from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image
import os
import json
import logging
from datetime import datetime
import torch
import evaluate
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Configure logging
log_filename = f"deplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

with open("val_data.json", "r") as f:
    data = json.load(f)
    
logger.info(f"Loaded {len(data)} items from val_data.json")

processor = Pix2StructProcessor.from_pretrained('./deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('./deplot')
model = model.to(device)  # Move model to GPU
model.eval()  # Set model to evaluation mode

logger.info("Model loaded and moved to device")

metric = evaluate.load("./metric")

single_metric = evaluate.load("./metric")

logger.info("Metric loaded !")

true_positives = 0
false_positives = 0
false_negatives = 0

for idx, item in enumerate(data):
    image_path = item["conversations"][0]["value"].replace("./DVQA/images/", "./test_images/")
    label = item["conversations"][1]["value"]
    
    logger.info(f"Processing image {idx+1}/{len(data)}: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model.generate(**inputs, max_new_tokens=512)
    
    result = processor.decode(predictions[0], skip_special_tokens=True)
    logger.info(f"Prediction: {result}")
    result = result.split("<0x0A>")[1:]
    result = "||".join(result)

    metric.add_batch(predictions=[result],references=[label])
    logger.info(f"Prediction formated: {result}")
    logger.info(f"Label: {label}")
    try:
        single_result = single_metric.compute(predictions=[result],references=[label])

        logger.info(f"single evaluate result:{single_result}")
        true_positives += single_result["true_positives"]
        false_positives += single_result["false_positives"]
        false_negatives += single_result["false_negatives"]
    except TypeError as e:
        print(e)

        

logger.info("Processing completed")
print(f"evaluate result:{metric.compute()}")

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

logging.info(f"precision:{precision}\t reacll:{recall}\tf1:{f1}")