from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import json
import evaluate
import logging
from datetime import datetime

log_filename = f"charttotable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # This will also print logs to console
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting DePlot processing...")

model_name = "khhuangchart-to-table"
model = VisionEncoderDecoderModel.from_pretrained(model_name).cuda()
processor = DonutProcessor.from_pretrained(model_name)
logger.info("Model loaded and moved to device")
input_prompt = "<data_table_generation> <s_answer>"

with open("val_data.json") as f:
    data = json.load(f)
logger.info(f"Loaded {len(data)} items from val_data.json")

metric = evaluate.load("./metric")
true_positives = 0
false_positives = 0
false_negatives = 0

for idx, item in enumerate(data):
    image_path = item["conversations"][0]["value"].replace("./DVQA/images/", "./test_images/")
    label = item["conversations"][1]["value"]
    
    logger.info(f"Processing image {idx+1}/{len(data)}: {image_path}")

    img = Image.open(image_path)
    logging.info("====================================================================")
    logger.info(f"Processing image {idx+1}/{len(data)}: {image_path}")
    pixel_values = processor(img.convert("RGB"), return_tensors="pt").pixel_values
    pixel_values = pixel_values.cuda()
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt", max_length=510).input_ids.cuda()#.squeeze(0)


    outputs = model.generate(
            pixel_values.cuda(),
            decoder_input_ids=decoder_input_ids.cuda(),
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
    extracted_table = sequence.split("<s_answer>")[1].strip()
    result = extracted_table.split("&&&")
    if "TITLE" in result[0]:
        result = result[1:]
    result = "||".join(result)
    logger.info(f"Prediction: {extracted_table}")
    # logger.info(f"Prediction formated: {result}")
    logger.info(f"Label: {label}")
    try:
        single_result = metric.compute(predictions=[result],references=[label])

        logger.info(f"single evaluate result:{single_result}")
        true_positives += single_result["true_positives"]
        false_positives += single_result["false_positives"]
        false_negatives += single_result["false_negatives"]
    except TypeError as e:
        print(e)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
logging.info("===================================================================================")
logging.info(f"precision:{precision}\t reacll:{recall}\tf1:{f1}")
logging.info("===================================================================================")