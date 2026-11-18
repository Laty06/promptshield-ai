"""
Évalue la baseline par règles sur le jeu de tests indépendant (custom_tests.csv).

Usage :
    python eval_custom_tests.py

Le fichier custom_tests.csv doit être dans le dossier data/ (ou à côté du script).
"""

import os

import pandas as pd

from baseline_rules import rule_based_predict

CSV_PATH = "data/custom_tests.csv" if os.path.exists("data/custom_tests.csv") else "custom_tests.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    df["pred"] = df["text"].apply(rule_based_predict)

    print(f"Fichier : {CSV_PATH} ({len(df)} cas)")
    print("\nRépartition :")
    print(df["type"].value_counts().to_string())

    # Recall par type d'injection (sur les injections seulement)
    print("\nRecall par type d'injection :")
    for t in ["direct", "indirect"]:
        sub = df[df["type"] == t]
        detected = int(sub["pred"].sum())
        print(f"  {t:9s}: {detected}/{len(sub)} détectées ({detected / len(sub):.0%})")

    # Faux positifs sur les messages normaux
    normal = df[df["type"] == "normal"]
    fp = int(normal["pred"].sum())
    print(f"\nFaux blocages sur messages normaux : {fp}/{len(normal)} ({fp / len(normal):.0%})")

    # Détail des erreurs, utile pour l'analyse d'erreurs du rapport
    errors = df[df["label"] != df["pred"]]
    print(f"\nErreurs ({len(errors)}) :")
    for _, row in errors.iterrows():
        kind = "FAUX POSITIF" if row["label"] == 0 else "INJECTION MANQUÉE"
        print(f"  [{row['id']:>2}] {kind} ({row['type']}) : {row['text'][:90]}")


if __name__ == "__main__":
    main()
