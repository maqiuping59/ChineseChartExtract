---
license: apache-2.0
language: en
---


# The Chart-To-Table Model

The Chart-To-Table model was introduced in the paper "Do LVLMs Understand Charts?  
Analyzing and Correcting Factual Errors in Chart Captioning" for converting a chart into a structured table. The generated tables use `&&&` to delimit rows and `|` to delimit columns. The underlying architecture of this model is UniChart.



### How to use


```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

model_name = "khhuang/chart-to-table"
model = VisionEncoderDecoderModel.from_pretrained(model_name).cuda()
processor = DonutProcessor.from_pretrained(model_name)

image_path = "PATH_TO_IMAGE"

# Format text inputs

input_prompt = "<data_table_generation> <s_answer>"

# Encode chart figure and tokenize text
img = Image.open(IMAGE_PATH)
pixel_values = processor(img.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values
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
```

### Citation
```bibtex
@misc{huang-etal-2023-do,
    title = "Do LVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning",
    author = "Huang, Kung-Hsiang  and
      Zhou, Mingyang and
      Chan, Hou Pong  and
      Fung, Yi R. and
      Wang, Zhenhailong and
      Zhang, Lingyu and
      Chang, Shih-Fu and
      Ji, Heng",
    year={2023},
    eprint={2312.10160},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}    
```