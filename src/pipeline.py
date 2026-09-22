"""Pembagian data latih/uji dan pelatihan Gaussian Naive Bayes.

Fungsi di modul ini dipakai identik untuk skenario dengan maupun tanpa
pembersihan data, sehingga perbandingan keduanya adil: satu-satunya
perbedaan antara kedua skenario adalah ada atau tidaknya tahap
pembersihan pada data yang masuk ke sini.
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

RANDOM_STATE = 42
TEST_SIZE = 0.2


def split_features_target(df, target_col="Outcome"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_and_predict(df, target_col="Outcome"):
    """Latih Gaussian Naive Bayes dan hasilkan prediksi pada data uji.

    Pembagian data memakai stratifikasi pada label agar proporsi kelas
    (sekitar 65:35) tetap terjaga baik pada data latih maupun data uji.
    random_state ditetapkan agar hasil dapat direproduksi.
    """
    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
    }
