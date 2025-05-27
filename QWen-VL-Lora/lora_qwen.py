import torch
from datasets import Dataset
from modelscope import  snapshot_download
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
import swanlab
import json
import os
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
from markdown_loss import MarkdownTableLoss
import evaluate

@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen-VL"
    model_path: str = "Qwen/Qwen-VL"
    data_path: str = "./train_chinese_chart.json"
    output_dir: str = "./output/"
    prompt: str = """你是一个智能助手，目的是从图表中抽取真实数据，并转为markdown形式。""",
    max_length: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    eval_batch_size: int = 4
    num_eval_samples: int = 100
    markdown_loss_weight: float = 0.2
    
    use_kl_loss: bool = False
    kl_loss_weight: float = 0.1
    use_l2_loss: bool = False
    l2_loss_weight: float = 0.01
    use_contrastive_loss: bool = False
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.1


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_split_dataset(config: TrainingConfig) -> tuple[Dataset, Dataset, Dataset]:
    with open(config.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    raw_data = raw_data[:10000] 
    dataset = Dataset.from_list(raw_data)
    

    first_split = dataset.train_test_split(train_size=0.9, test_size=0.1)
    train_val_dataset = first_split['train']
    test_dataset = first_split['test']
    
  
    second_split = train_val_dataset.train_test_split(train_size=0.78, test_size=0.22)  # 7/9:2/9
    train_dataset = second_split['train']
    val_dataset = second_split['test']
    
    return train_dataset, val_dataset, test_dataset

class DataProcessor:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(config.model_path)
        
    def process_func(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        conversation = example["conversations"]
        input_content = conversation[0]["value"]
        output_content = conversation[1]["value"]
        
 
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{input_content}",
                        "resized_height": 448,
                        "resized_width": 448,
                    },
                    {"type": "text", "text": self.config.prompt},
                ],
            }
        ]
        

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        

        inputs = {key: value.tolist() for key, value in inputs.items()}
        

        response = self.tokenizer(
            f"{output_content}", 
            add_special_tokens=False,
            padding=False,
            truncation=False
        )
        

        input_ids = (
            inputs["input_ids"][0] + 
            response["input_ids"] + 
            [self.tokenizer.pad_token_id]
        )
        attention_mask = (
            inputs["attention_mask"][0] + 
            response["attention_mask"] + 
            [1]
        )
        labels = (
            [-100] * len(inputs["input_ids"][0]) +
            response["input_ids"] +
            [self.tokenizer.pad_token_id]
        )
        

        if len(input_ids) > self.config.max_length:
            input_ids = input_ids[:self.config.max_length]
            attention_mask = attention_mask[:self.config.max_length]
            labels = labels[:self.config.max_length]
            

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),  
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long), 
            "labels": torch.tensor(labels, dtype=torch.long), 
            "pixel_values": torch.tensor(inputs['pixel_values'], dtype=torch.float32, requires_grad=True), 
            "image_grid_thw": torch.tensor(inputs['image_grid_thw'], dtype=torch.long).squeeze(0) 
        }

    def predict(self, message: List[Dict[str, Any]], model: torch.nn.Module) -> str:
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.config.max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

def setup_logging(config: TrainingConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config.output_dir, 'training.log'))
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

class CustomTrainer(Trainer):
    def __init__(self, *args, markdown_loss_weight=0.2, **kwargs):
        self.markdown_loss_weight = markdown_loss_weight
        self.processor = kwargs.pop('processor', None)
        
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.markdown_loss = MarkdownTableLoss(text_weight=0.6, numeric_weight=0.4)

    def get_processor(self):
        if hasattr(self, 'processing_class') and self.processing_class is not None:
            return self.processing_class
        elif hasattr(self, 'processor') and self.processor is not None:
            return self.processor
        elif hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.logger.warning("Using deprecated tokenizer attribute")
            return self.tokenizer
        return None

    def safe_convert_tokens(self, tokens):
        try:

            tokens_cpu = tokens.cpu().numpy()

            processor = self.get_processor()
            vocab_size = None
            
            if processor is not None:
                if hasattr(processor, 'vocab_size'):
                    vocab_size = processor.vocab_size
                elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'vocab_size'):
                    vocab_size = processor.tokenizer.vocab_size
            
            if vocab_size is not None:

                tokens_cpu = np.clip(tokens_cpu, 0, vocab_size - 1)
            
   
            return torch.tensor(tokens_cpu, device=tokens.device)
        except Exception as e:
            self.logger.warning(f"Error in token conversion: {str(e)}")
            return tokens

    def decode_tokens(self, tokens, skip_special_tokens=True):
        try:

            safe_tokens = self.safe_convert_tokens(tokens)
            processor = self.get_processor()
            
            if processor is not None:
                try:
                    decoded = processor.batch_decode(safe_tokens, skip_special_tokens=skip_special_tokens)
                except Exception as e:
                    self.logger.warning(f"Primary decode failed: {str(e)}")
                    if hasattr(processor, 'tokenizer'):
                        decoded = processor.tokenizer.batch_decode(safe_tokens, skip_special_tokens=skip_special_tokens)
                    else:
                        raise e
            else:
                self.logger.warning("No processor found")
                return None

            if decoded and isinstance(decoded[0], str) and decoded[0].strip():
                # 清理解码后的文本
                cleaned_text = decoded[0].strip().replace('\x00', '')
                if cleaned_text:
                    return cleaned_text
            return None
        except Exception as e:
            self.logger.warning(f"Error in token decoding: {str(e)}")
            return None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits


        labels = inputs.get("labels")
        

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
   
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        base_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        try:
 
            pred_tokens = torch.argmax(shift_logits, dim=-1)
            pred_markdown = self.decode_tokens(pred_tokens)
            

            valid_labels = torch.where(shift_labels == -100, 
                                     torch.zeros_like(shift_labels), 
                                     shift_labels)
            target_markdown = self.decode_tokens(valid_labels)
            
            if pred_markdown is not None and target_markdown is not None:
                # 确保文本包含markdown表格格式
                if '|' in pred_markdown and '|' in target_markdown:
                    try:
                        markdown_table_loss = self.markdown_loss(pred_markdown, target_markdown)
                        total_loss = (1 - self.markdown_loss_weight) * base_loss + self.markdown_loss_weight * markdown_table_loss
                        
                        return (total_loss, outputs) if return_outputs else total_loss
                    except Exception as e:
                        self.logger.warning(f"Error computing markdown table loss: {str(e)}")
                else:
                    self.logger.warning("No table format found in decoded text")
            else:
                self.logger.warning("Failed to decode tokens")
        except Exception as e:
            self.logger.warning(f"Error in loss computation: {str(e)}")

        return (base_loss, outputs) if return_outputs else base_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.logger.info(f"Training loss: {loss.item():.4f}")
        return loss

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        super().log(logs, start_time)
        for key, value in logs.items():
            self.logger.info(f"{key}: {value}")

def evaluate_model(model: torch.nn.Module, val_dataset: Dataset, processor: DataProcessor, config: TrainingConfig) -> Dict[str, float]:
    logger = logging.getLogger(__name__)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    accuracy_metric = evaluate.load("maqiuping59/table_markdown")
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for i in tqdm(range(min(len(val_dataset), config.num_eval_samples)), desc="Evaluating"):
            
            item = val_dataset[i]
            print(item)
            logger.debug(f"Processing item: {item}")
            
            if not isinstance(item, dict):
                logger.warning(f"Unexpected item type: {type(item)}")
                continue
            
            conversations = item.get("conversations", [])
            if not conversations or len(conversations) < 2:
                logger.warning(f"Invalid conversations format in item {i}")
                continue
            
            image_file_path = conversations[0].get("value", "")
            label = conversations[1].get("value", "")
            
            if not image_file_path or not label:
                logger.warning(f"Missing image path or label in item {i}")
                continue
            
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file_path,
                        "resized_height": 448,
                        "resized_width": 448,
                    },
                    {"type": "text", "text": config.prompt}
                ]
            }]
            
            prediction = processor.predict(messages, model)
            print(prediction)
            all_predictions.append(prediction)
            all_labels.append(label)

    
    if not all_predictions:
        logger.warning("No valid predictions were made during evaluation")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "num_samples": 0
        }
    
    metrics = accuracy_metric.compute(predictions=all_predictions, references=all_labels)
    metrics["num_samples"] = len(all_predictions)
    
    # Log metrics
    logger.info("Evaluation Results:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric_name}: {value:.4f}")
        else:
            logger.info(f"{metric_name}: {value}")
    
    swanlab.log({
        "eval/precision": metrics["precision"],
        "eval/recall": metrics["recall"],
        "eval/f1": metrics["f1"],
        "eval/true_positives": metrics["true_positives"],
        "eval/false_positives": metrics["false_positives"],
        "eval/false_negatives": metrics["false_negatives"],
        "eval/num_samples": metrics["num_samples"]
    })
    
    return metrics

def select_best_checkpoint(output_dir: str, model: torch.nn.Module, val_dataset: Dataset, 
                         processor: DataProcessor, config: TrainingConfig) -> Optional[str]:
    logger = logging.getLogger(__name__)
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    
    if not checkpoints:
        logger.warning("No checkpoints found.")
        return None
    
    best_checkpoint = None
    best_metric = float('-inf')
    best_metrics = None
    

    val_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        checkpoint_path = os.path.join(output_dir, checkpoint)
        logger.info(f"Evaluating checkpoint: {checkpoint}")
        
        if hasattr(model, 'peft_config'):
            if hasattr(model, 'disable_adapter'):
                model.disable_adapter()
            delattr(model, 'peft_config')
        
  
        checkpoint_model = PeftModel.from_pretrained(
            model,
            model_id=checkpoint_path,
            config=val_lora_config
        )
        logger.info(f"Successfully loaded checkpoint from {checkpoint}")
        
        metrics = evaluate_model(checkpoint_model, val_dataset, processor, config)
        current_metric = metrics["f1"]
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_checkpoint = checkpoint_path
            best_metrics = metrics
            logger.info(f"New best checkpoint found: {checkpoint}")
            logger.info(f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        

    

    if best_metrics:
        swanlab.log({
            "best_checkpoint/f1": best_metrics["f1"],
            "best_checkpoint/precision": best_metrics["precision"],
            "best_checkpoint/recall": best_metrics["recall"],
            "best_checkpoint/true_positives": best_metrics["true_positives"],
            "best_checkpoint/false_positives": best_metrics["false_positives"],
            "best_checkpoint/false_negatives": best_metrics["false_negatives"]
        })
    
    return best_checkpoint

def parse_markdown_table(table_str: str):
    lines = table_str.strip().split('\n')
    if len(lines) < 3:  
        return None, None
    
 
    header = [col.strip() for col in lines[0].strip('|').split('|')]
    

    data_rows = []
    for line in lines[2:]:  
        if line.strip():
            row = [cell.strip() for cell in line.strip('|').split('|')]
            data_rows.append(row)
            
    return header, data_rows

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a Qwen-VL model with LoRA")
    parser.add_argument("--model_path", type=str, default="./Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data_path", type=str, default="./train_data.json")
    parser.add_argument("--output_dir", type=str, default="./output/Qwen2-VL-TableExactor-OCR-2B")
    parser.add_argument("--prompt", type=str, default="你是一个智能助手,目标是提取图表中的信息,并将数据转换为markdown形式的表格(用‘|’分隔单元格，用‘|--|’分割标题行)。")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--markdown_loss_weight", type=float, default=0.8)
    
    args = parser.parse_args()
    return TrainingConfig(**vars(args))

def main():
    # Parse arguments and setup configuration
    prompt =  """
                你是一个精确的图表助手。请仔细分析图片中的表格，并完成以下任务：
                1. 准确识别表格的行列标题
                2. 精确提取表格中的所有数值
                3. 按照原始格式转换成markdown表格
                4. 确保数值的精确性，包括小数点位数
                请确保输出的表格与原始图片中的内容完全一致。
            """
    config = parse_args()
    config.prompt = prompt
    logger = setup_logging(config)
    logger.info("Starting training with configuration:")
    logger.info(str(config))
    
    set_seed()
    
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(config)
    logger.info(f"Dataset split complete:")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    data_processor = DataProcessor(config)
    train_data = train_dataset.map(data_processor.process_func)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=torch.float16,
        use_cache=False  
    )
    
    if hasattr(model, 'peft_config'):
        logger.info("Removing existing peft configuration")
        if hasattr(model, 'disable_adapter'):
            model.disable_adapter()
        delattr(model, 'peft_config')
    
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    
    train_peft_model = get_peft_model(model, lora_config)
    logger.info("Created new PEFT model with LoRA configuration")
    
    # Configure training arguments with evaluation
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=10,
        logging_first_step=10,
        num_train_epochs=config.num_train_epochs,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        learning_rate=config.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        do_eval=True,
        use_cpu=not torch.cuda.is_available(),
        fp16=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
    )
    
    # Configure SwanLab callback
    swanlab_callback = SwanLabCallback(
        project="Qwen2-VL-ChartExtractor",
        experiment_name="7B-1kdata",
        config={
            "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct",
            "dataset": "DVQA samples",
            "model_id": "Qwen-VL-2b-ChartExtractor",
            "output_dir": config.output_dir,
            "prompt": config.prompt,
            "train_data_number": len(train_data),
            "token_max_length": config.max_length,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        },
    )
    
    # Initialize trainer with progress bar and validation dataset
    trainer = CustomTrainer(
        model=train_peft_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_dataset.map(data_processor.process_func),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=data_processor.tokenizer,
            padding=True
        ),
        callbacks=[swanlab_callback],
        markdown_loss_weight=config.markdown_loss_weight,
        processor=data_processor.processor
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed")
    
    # Select best checkpoint and perform final evaluation
    logger.info("Selecting best checkpoint...")
    best_checkpoint = select_best_checkpoint(config.output_dir, model, val_dataset, data_processor, config)
    
    if best_checkpoint:
        logger.info(f"Best checkpoint found: {best_checkpoint}")
        val_peft_model = PeftModel.from_pretrained(
            model,
            model_id=best_checkpoint,
            config=lora_config
        )
        
        # Final evaluation on test set
        logger.info("Performing final evaluation on test set...")
        final_metrics = evaluate_model(val_peft_model, test_dataset, data_processor, config)
        logger.info("Final test metrics:")
        logger.info(str(final_metrics))
        
        # Log predictions for a few test samples
        test_image_list = []
        test_metrics = evaluate.load("maqiuping59/table_markdown")
        for i in tqdm(range(min(10, len(test_dataset))), desc="Generating test predictions"):
            try:
                item = test_dataset[i]
                # 确保item是字典类型
                if not isinstance(item, dict):
                    logger.warning(f"Unexpected item type: {type(item)}")
                    continue
                
                # 获取图像路径和标签
                conversations = item.get("conversations", [])
                if not conversations or len(conversations) < 2:
                    logger.warning(f"Invalid conversations format in item {i}")
                    continue
                
                image_file_path = conversations[0].get("value", "")
                label = conversations[1].get("value", "")
                
                if not image_file_path or not label:
                    logger.warning(f"Missing image path or label in item {i}")
                    continue
                
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file_path,
                            "resized_height": 448,
                            "resized_width": 448,
                        },
                        {"type": "text", "text": config.prompt}
                    ]
                }]
                
                response = data_processor.predict(messages, val_peft_model)
                logger.info(f"Prediction: {response}")
                logger.info(f"Ground truth: {label}\n")
                test_metrics.add(predictions=[response], references=[label])

                
                test_image_list.append(swanlab.Image(image_file_path, caption=response))
            except Exception as e:
                logger.error(f"Error processing test sample {i}: {str(e)}")
                continue
        print(test_metrics.compute())
        
        if test_image_list:
            swanlab.log({"Test_Predictions": test_image_list})
    
    swanlab.finish()
    logger.info("Training session completed")

if __name__ == "__main__":
    main()




