import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import json
import os
from sklearn.preprocessing import label_binarize


def calculate_recall(labels, preds, class_id=None):
    """
    Calculate recall for a specified class or for the positive label in a multi-class task.

    Args:
    labels (np.array): Array of true labels.
    preds (np.array): Array of predicted labels.
    class_id (int or None): Class ID for which to calculate recall. If None, calculate recall for the positive label.

    Returns:
    float: Recall for the specified class or for the positive label.
    """
    if class_id is not None:
        true_class = labels == class_id
        predicted_class = preds == class_id
    else:
        true_class = labels > 0
        predicted_class = preds > 0

    true_positives = np.sum(true_class & predicted_class)
    false_negatives = np.sum(true_class & ~predicted_class)

    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    return recall


class Metrics:
    def __init__(self, header="Main", num_class=2):
        self.num_class = num_class
        self.header = header
        self.reset()

    def reset(self):
        self.accuracy = 0
        self.auc = 0
        self.recall = {
            "pos": 0,
            # 根据num_class的值，初始化recall的key，例如有两类就是recall for class 0, recall for class 1
            # Recall for each class
            **{f"class_{i}": 0 for i in range(self.num_class)}

        }

    def update(self, predictions, targets):
        # Update accuracy
        self.accuracy = accuracy_score(targets, predictions)

        # Convert predictions to one-hot format for AUC calculation
        if self.num_class > 2:
            targets_one_hot = label_binarize(
                targets, classes=range(self.num_class))
            predictions_one_hot = label_binarize(
                predictions, classes=range(self.num_class))
        else:
            targets_one_hot = targets
            # For binary classification, use the labels directly
            predictions_one_hot = predictions

        # Update AUC; handle binary and multiclass scenarios
        if self.num_class == 2:
            # Directly compute AUC for binary classification
            self.auc = roc_auc_score(targets_one_hot, predictions_one_hot)
        else:
            # Compute AUC using a one-vs-rest approach for multiclass classification
            try:
                self.auc = roc_auc_score(
                    targets_one_hot, predictions_one_hot, multi_class='ovr')
            except ValueError as e:
                print(f"Failed to calculate AUC: {e}")
                self.auc = None

        # Calculate recall_pos for all positive classes
        self.recall["pos"] = recall_score(
            targets, predictions, average='binary')
        for i in range(self.num_class):
            self.recall[f"class_{i}"] = calculate_recall(
                targets, predictions, class_id=i)

    def __str__(self):
        return f"[{self.header}] Acc: {self.accuracy:.4f}, Auc: {self.auc:.4f}, Recall: {self.recall['pos']:.4f}" +\
            "".join(
                [f", Recall_{i}: {self.recall[f'class_{i}']:.4f}" for i in range(self.num_class)])

    def _store(self, param, save_path):
        # Prepare the result dictionary
        res = {
            "accuracy": round(self.accuracy, 4),
            "auc": round(self.auc, 4),
            "recall": {
                "pos": round(self.recall["pos"], 4),
                **{f"class_{i}": round(self.recall[f'class_{i}'], 4) for i in range(self.num_class)
                   }}
        }

        # Check if the file exists and load its content if it does
        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        # Append the new data
        existing_data.append({
            "result": res,
            "param": param
        })

        # Save the updated data back to the file
        with open(save_path, 'w') as file:
            json.dump(existing_data, file, indent=4)


if __name__ == "__main__":
    pass
