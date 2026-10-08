"""Evaluasi performa model: confusion matrix dan metrik klasifikasi
standar untuk klasifikasi multi-kelas (akurasi, precision, recall,
F1-score makro, serta rincian per kelas).

Generik terhadap nama dan jumlah kelas -- dipakai untuk label
InsulinResistance (3 kelas: Rendah/Sedang/Tinggi) maupun label lain.
Saat `labels` diberikan secara eksplisit, confusion matrix dan metrik
per kelas selalu memakai daftar itu (bukan hanya kelas yang muncul di
data), sehingga kelas yang tidak pernah muncul pada suatu skenario
tetap terlihat eksplisit sebagai baris/kolom kosong, bukan diam-diam
hilang dari laporan.
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_test, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_test) | set(y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    laporan = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "akurasi": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(
            precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(
            f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0), 4
        ),
        "per_kelas": {
            kelas: {
                "precision": round(laporan[kelas]["precision"], 4),
                "recall": round(laporan[kelas]["recall"], 4),
                "f1_score": round(laporan[kelas]["f1-score"], 4),
                "support": int(laporan[kelas]["support"]),
            }
            for kelas in labels
        },
    }
