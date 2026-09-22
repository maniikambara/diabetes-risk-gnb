"""Evaluasi performa model: confusion matrix dan metrik klasifikasi
standar (akurasi, precision, recall, F1-score).

Precision, recall, dan F1-score dihitung terhadap kelas positif
(Outcome = 1, pasien berisiko diabetes), karena kelas inilah yang
relevan secara klinis untuk skenario skrining.
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "confusion_matrix": cm.tolist(),
        "akurasi": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=1), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label=1), 4),
        "f1_score": round(f1_score(y_test, y_pred, pos_label=1), 4),
    }
