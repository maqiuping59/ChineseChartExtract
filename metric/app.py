import evaluate
import gradio as gr
import os



module = evaluate.load("maqiuping59/table_markdown")

def compute_metrics(predictions, references):
    results = module.compute(predictions=predictions, references=references)
    return results

# 创建界面
demo = gr.Interface(
    fn=compute_metrics,
    inputs=[
        gr.Textbox(label="Predictions (Markdown Table)", lines=10),
        gr.Textbox(label="References (Markdown Table)", lines=10)
    ],
    outputs=gr.JSON(label="Results"),
    title="Table Markdown Metrics",
    description="Evaluate the accuracy of table data extraction or generation by comparing predicted tables with reference tables.",
    examples=[
        [
            "|A|B|\n|1|2|",
            "|A|B|\n|1|3|"
        ],
        [
            "|  | lobby | search | band |\n|--|-------|--------|------|\n| desire | 5 | 8 | 7 |\n| wage | 1 | 5 | 3 |",
            "|  | lobby | search | band |\n|--|-------|--------|------|\n| desire | 5 | 8 | 7 |\n| wage | 1 | 5 | 3 |"
        ]
    ]
)

demo.launch()
