"""
PromptShield AI - Modèle ML : TF-IDF + SVM.

Étapes :
  1. Charger le dataset deepset (train/test officiels).
  2. Vérifier qu'aucun de vos cas de test n'est dans le train (pas de data leakage).
  3. Entraîner TF-IDF + SVM sur le train SEULEMENT.
  4. Évaluer sur le test officiel, puis sur vos 60 cas (data/custom_tests.csv).
  5. Sauvegarder le modèle dans models/svm_tfidf.joblib.

Usage :
    python train_svm.py
"""

import os
import time

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

CSV_PATH = "data/custom_tests.csv" if os.path.exists("data/custom_tests.csv") else "custom_tests.csv"
MODEL_PATH = "models/svm_tfidf.joblib"


def build_model():
    """
    Pipeline en 2 blocs :
      - TfidfVectorizer : transforme chaque texte en un vecteur de nombres.
        ngram_range=(1, 2) regarde les mots seuls ET les paires de mots
        (par exemple "ignore previous"), ce qui aide à repérer les injections.
      - LinearSVC : le SVM, qui trace une frontière entre "normal" (0)
        et "injection" (1).
    """
    return make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
        LinearSVC(C=1.0, class_weight="balanced", random_state=42),
    )


def check_no_leakage(train_texts, custom_df):
    """Vérifie qu'aucun cas de test n'apparaît à l'identique dans le train."""
    train_set = {t.strip().lower() for t in train_texts}
    overlap = [t for t in custom_df["text"] if t.strip().lower() in train_set]
    if overlap:
        print(f"ATTENTION : {len(overlap)} cas de test sont aussi dans le train !")
        for t in overlap:
            print("  -", t[:80])
    else:
        print("Vérification data leakage : OK (aucun cas de test dans le train).")


def evaluate_deepset(model, test):
    texts, y_true = test["text"], test["label"]
    y_pred = model.predict(texts)
    print("\n=== Test officiel deepset (%d messages) ===" % len(texts))
    print("Matrice de confusion [[VN, FP], [FN, VP]] :")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["normal", "injection"]))


def evaluate_custom(model, df):
    df = df.copy()
    df["pred"] = model.predict(df["text"])

    print("\n=== Vos %d cas de test indépendants ===" % len(df))
    print("Recall par type d'injection :")
    for t in ["direct", "indirect"]:
        sub = df[df["type"] == t]
        detected = int(sub["pred"].sum())
        print(f"  {t:9s}: {detected}/{len(sub)} détectées ({detected / len(sub):.0%})")

    normal = df[df["type"] == "normal"]
    fp = int(normal["pred"].sum())
    print(f"Faux blocages sur messages normaux : {fp}/{len(normal)} ({fp / len(normal):.0%})")

    tp = int(df[(df["label"] == 1) & (df["pred"] == 1)].shape[0])
    alerts = int(df["pred"].sum())
    if alerts:
        print(f"Precision sur vos cas : {tp}/{alerts} ({tp / alerts:.0%})")

    errors = df[df["label"] != df["pred"]]
    print(f"\nErreurs ({len(errors)}) :")
    for _, row in errors.iterrows():
        kind = "FAUX POSITIF" if row["label"] == 0 else "INJECTION MANQUÉE"
        print(f"  [{row['id']:>2}] {kind} ({row['type']}) : {row['text'][:90]}")


def measure_latency(model, texts):
    """Temps moyen pour prédire UN message (en millisecondes)."""
    start = time.perf_counter()
    for t in texts:
        model.predict([t])
    elapsed = time.perf_counter() - start
    print(f"\nLatence moyenne : {1000 * elapsed / len(texts):.2f} ms par message")


def main():
    ds = load_dataset("deepset/prompt-injections")
    train, test = ds["train"], ds["test"]
    custom = pd.read_csv(CSV_PATH)

    check_no_leakage(train["text"], custom)

    # Entraînement : uniquement sur le train
    model = build_model()
    model.fit(train["text"], train["label"])
    print(f"Modèle entraîné sur {len(train['text'])} exemples.")

    evaluate_deepset(model, test)
    evaluate_custom(model, custom)
    measure_latency(model, list(custom["text"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
