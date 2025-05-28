---
title: Table Markdown Metrics
emoji: 📊 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
- table
- markdown
description: >-
  Table evaluation metrics for assessing the matching degree between predicted and reference tables.
  It calculates precision, recall, and F1 score for table data extraction or generation tasks.
---

# Metric Card for Table Markdown Metrics

## Metric Description

This metric evaluates the accuracy of table data extraction or generation by comparing predicted tables with reference tables. It calculates:

1. Precision: The ratio of correctly predicted cells to the total number of cells in the predicted table
2. Recall: The ratio of correctly predicted cells to the total number of cells in the reference table
3. F1 Score: The harmonic mean of precision and recall

## How to Use

This metric requires predictions and references as inputs in Markdown table format.

```python
>>> table_metric = evaluate.load("table_markdown")
>>> results = table_metric.compute(
...     predictions="|  | lobby | search | band | charge | chain ||--|--|--|--|--|--|| desire | 5 | 8 | 7 | 5 | 9 || wage | 1 | 5 | 3 | 8 | 5 |",
...     references="|  | lobby | search | band | charge | chain ||--|--|--|--|--|--|| desire | 1 | 6 | 7 | 5 | 9 || wage | 1 | 5 | 2 | 8 | 5 |"
... )
>>> print(results)
{'precision': 0.7, 'recall': 0.7, 'f1': 0.7, 'true_positives': 7, 'false_positives': 3, 'false_negatives': 3}
```

### Inputs
- **predictions** (`str`): Predicted table in Markdown format.
- **references** (`str`): Reference table in Markdown format.

### Output Values
- **precision** (`float`): Precision score. Range: [0,1]
- **recall** (`float`): Recall score. Range: [0,1]
- **f1** (`float`): F1 score. Range: [0,1]
- **true_positives** (`int`): Number of correctly predicted cells
- **false_positives** (`int`): Number of incorrectly predicted cells
- **false_negatives** (`int`): Number of cells that were not predicted

### Examples

Example  - Complex table comparison:
```python
>>> table_metric = evaluate.load("table_markdown")
>>> results = table_metric.compute(
...     predictions="""
... |  | lobby | search | band |
... |--|-------|--------|------|
... | desire | 5 | 8 | 7 |
... | wage | 1 | 5 | 3 |
... """,
...     references="""
... |  | lobby | search | band |
... |--|-------|--------|------|
... | desire | 5 | 8 | 7 |
... | wage | 1 | 5 | 3 |
... """
... )
>>> print(results)
{'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'true_positives': 6, 'false_positives': 0, 'false_negatives': 0}
```

## Limitations and Bias

1. The metric assumes that tables are well-formed in Markdown format
2. The comparison is case-sensitive
3. The metric does not handle merged cells or complex table structures
4. The metric treats each cell as a separate unit and does not consider the semantic meaning of the content

## Citation(s)
```bibtex
@article{scikit-learn,
  title={Research on Chinese Chart Data Extraction Methods},
  author={Qiuping Ma,Hangshuo Bi,Qi Zhang,Xiaofan Zhao},
  journal={None},
  volume={0},
  pages={0--0},
  year={2025}
}
```

## Further References

- [Markdown Tables](https://www.markdownguide.org/extended-syntax/#tables)
- [Table Structure Recognition](https://paperswithcode.com/task/table-structure-recognition)
- [Table Extraction](https://paperswithcode.com/task/table-extraction)
