import os
import json
import logging
from datetime import datetime
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, re
import evaluate

# 设置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"unichart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

model_name = "ahmed-masry/unichart-base-960"

input_prompt = "<extract_data_table> <s_answer>"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)


with open("../val_data.json","r") as f:
    data = json.load(f)


metric = evaluate.load("./table_metric")
new_metric = evaluate.load("./metric")
true_positives = 0
false_positives = 0
false_negatives = 0

for item in data:
    image_path = item["conversations"][0]["value"].replace("./DVQA/images/","../test_images/")
    logger.info(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    ground_truth = item["conversations"][1]["value"]
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    prediction_sequence = sequence.split("<s_answer>")[1].strip()
    sequence = prediction_sequence.replace("Characteristic","").replace(" ","")
    sequence = sequence.split("&")
    head = sequence[0]
    if len(head.split("|")) == 2:
        labels = [i.replace(" ","").split("|")[0] for i in sequence[1:]]
        try:
            values = [i.replace(" ","").split("|")[1] for i in sequence[1:]]
        except:
            values = [i.replace(" ","").split("|")[0] for i in sequence[1:]]
        labels_str = "|".join(labels)+"|\n"
        values_str = "|".join(values)
        split_row = "|"+"--|"*len(labels)+"\n|"
        prediction_str = "|"+labels_str+split_row+values_str+"|"
    else:
        labels = head.split("|")[1:]
        labels = [i for i in labels if i is not None]
        legned = [i.replace(" ","").split("|")[0] for i in sequence[1:]]
        split_row = "|"+"--|"*len(legned)+"\n"
        prediction_str = "||"+"|".join(legned)+"|\n"+split_row
        rows = ""
        for i in range(len(labels)):
            row = "|"+labels[i]+"|"
            for j in range(len(legned)):
                try:
                    row += sequence[1+j].split("|")[i+1]+"|"
                except:
                    row += "|"
            rows += row+"\n"
        prediction_str += rows
    
    metric.add_batch(predictions=[prediction_str], references=[ground_truth])
    try:
        preevaluste = new_metric.compute(predictions=[prediction_str], references=[ground_truth])
        true_positives += preevaluste["true_positives"]
        false_positives += preevaluste["false_positives"]
        false_negatives += preevaluste["false_negatives"]
    except:
        pass

    logger.info(f"Image name: {os.path.basename(image_path)}")
    logger.info(f"Original prediction: {prediction_sequence}")
    logger.info(f"Formulated prediction: {prediction_str}")
    logger.info(f"Ground truth: {ground_truth}")
    logger.info("=" * 50)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

metrics_result = metric.compute()
logger.info(f"Metrics: {metrics_result}")
logger.info(f"Precision: {precision:.4f}\tRecall: {recall:.4f}\tF1: {f1:.4f}")
