import re
import json
import evaluate
import datasets

_DESCRIPTION = """
Table evaluation metrics for assessing the matching degree between predicted and reference tables. It calculates the following metrics:

1. Precision: The ratio of correctly predicted cells to the total number of cells in the predicted table
2. Recall: The ratio of correctly predicted cells to the total number of cells in the reference table
3. F1 Score: The harmonic mean of precision and recall

These metrics help evaluate the accuracy of table data extraction or generation.
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`str`): Predicted table in Markdown format.
    references (`str`): Reference table in Markdown format.

Returns:
    dict: A dictionary containing the following metrics:
        - precision (`float`): Precision score, range [0,1]
        - recall (`float`): Recall score, range [0,1]
        - f1 (`float`): F1 score, range [0,1]
        - true_positives (`int`): Number of correctly predicted cells
        - false_positives (`int`): Number of incorrectly predicted cells
        - false_negatives (`int`): Number of cells that were not predicted

Examples:
    >>> accuracy_metric = evaluate.load("accuracy")
    >>> results = accuracy_metric.compute(
    ...     predictions="|  | lobby | search | band | charge | chain ||--|--|--|--|--|--|| desire | 5 | 8 | 7 | 5 | 9 || wage | 1 | 5 | 3 | 8 | 5 |",
    ...     references="|  | lobby | search | band | charge | chain ||--|--|--|--|--|--|| desire | 1 | 6 | 7 | 5 | 9 || wage | 1 | 5 | 2 | 8 | 5 |"
    ... )
    >>> print(results)
    {'precision': 0.7, 'recall': 0.7, 'f1': 0.7, 'true_positives': 7, 'false_positives': 3, 'false_negatives': 3}
"""


_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


    
@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )
    def _extract_markdown_table(self,text):
        text = text.replace('\n', '')
        text = text.replace(" ","")
        pattern = r'\|(?:[^|]+\|)+[^|]+\|'
        matches = re.findall(pattern, text)
        
        if matches:
            return ''.join(matches)
        
        return None
    
    def _table_to_dict(self,table_str):
        result_dict = {}

        table_str = table_str.lstrip("|").rstrip("|")
        parts = table_str.split('||')
        parts = [part for part in parts if "--" not in part]
        legends = parts[0].split("|")

        rows = len(parts)
        if rows == 2:
            nums = parts[1].split("|")
            for i in range(len(nums)):
                result_dict[legends[i]]=float(nums[i])
        elif rows >=3:
            for i in range(1,rows):
                pre_row = parts[i]
                pre_row = pre_row.split("|")
                label = pre_row[0]
                result_dict[label] = {}
                for j in range(1,len(pre_row)):
                    result_dict[label][legends[j-1]] = float(pre_row[j])
        else:
            return None
        
        return result_dict
    def _markdown_to_dict(self,markdown_str):
        table_str = self._extract_markdown_table(markdown_str)
        if table_str:
            return self._table_to_dict(table_str)
        else:
            return None
    
    def _calculate_table_metrics(self,pred_table, true_table):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # 遍历预测表格的所有键值对
        for key, pred_value in pred_table.items():
            if key in true_table:
                true_value = true_table[key]
                if isinstance(pred_value, dict) and isinstance(true_value, dict):
                    nested_metrics = self._calculate_table_metrics(pred_value, true_value)
                    true_positives += nested_metrics['true_positives']
                    false_positives += nested_metrics['false_positives']
                    false_negatives += nested_metrics['false_negatives']
                # 如果值相等
                elif pred_value == true_value:
                    true_positives += 1
                else:
                    false_positives += 1
                    false_negatives += 1
            else:
                false_positives += 1

        # 计算未匹配的真实值
        for key in true_table:
            if key not in pred_table:
                false_negatives += 1

        # 计算指标
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def _compute(self, predictions, references):
        predictions = "".join(predictions)
        references = "".join(references)
        return self._calculate_table_metrics(self._markdown_to_dict(predictions), self._markdown_to_dict(references))
    

def main():
    accuracy_metric = Accuracy()

    # 计算指标
    results = accuracy_metric.compute(
        predictions=["""
|  | lobby | search | band | charge | chain ||--|--|--|--|--|--|| desire | 5 | 8 | 7 | 5 | 9 || wage | 1 | 5 | 3 | 8 | 5 |
"""],  # 预测的表格
        references=["""
|  | lobby | search | band | charge | chain ||--|--|--|--|--|--|| desire | 1 | 6 | 7 | 5 | 9 || wage | 1 | 5 | 2 | 8 | 5 |
"""],   # 参考的表格
    )
    print(results)  # 输出结果

if __name__ == '__main__':
    main()



