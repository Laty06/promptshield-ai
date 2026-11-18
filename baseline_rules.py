"""
PromptShield AI - Baseline par règles.

Idée : si le message contient une phrase typique d'injection, on le bloque.
C'est simple, mais facile à contourner : c'est justement ce qu'on veut
montrer en comparant avec le modèle ML plus tard.
"""

import re

from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

# 1) Liste de motifs suspects (expressions régulières, insensibles à la casse)
SUSPICIOUS_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|above|prior) (instructions|prompts?)",
    r"disregard (all |any )?(the )?(previous|above|prior)",
    r"forget (all |any )?(your |the )?(previous |prior )?(instructions|rules)",
    r"you are now",
    r"act as",
    r"pretend (to be|you are)",
    r"reveal (your |the )?(system )?prompt",
    r"system prompt",
    r"developer mode",
    r"jailbreak",
    # quelques motifs en français et en allemand (le dataset en contient)
    r"ignore[rz]? (toutes )?(les )?(instructions|consignes) (précédentes|ci-dessus)",
    r"oublie[rz]? (tes|vos) (instructions|consignes)",
    r"ignoriere (alle )?(vorherigen|obigen)",
    r"vergiss (alles|alle)",
]
COMPILED = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]


def rule_based_predict(text: str) -> int:
    """Retourne 1 (injection) si un motif suspect est trouvé, sinon 0 (normal)."""
    return int(any(p.search(text) for p in COMPILED))


def main():
    # 2) Chargement du dataset (train/test officiels)
    ds = load_dataset("deepset/prompt-injections")
    test = ds["test"]
    texts = test["text"]
    y_true = test["label"]  # 1 = injection, 0 = normal

    # 3) Prédiction par les règles
    y_pred = [rule_based_predict(t) for t in texts]

    # 4) Évaluation
    print(f"Taille du jeu de test : {len(texts)}")
    print("\nMatrice de confusion [[VN, FP], [FN, VP]] :")
    print(confusion_matrix(y_true, y_pred))
    print("\nRapport :")
    print(classification_report(y_true, y_pred, target_names=["normal", "injection"]))

    # 5) Exemples d'injections manquées (utile pour l'analyse d'erreurs du rapport)
    print("\nInjections manquées (faux négatifs), 5 premiers exemples :")
    missed = [t for t, yt, yp in zip(texts, y_true, y_pred) if yt == 1 and yp == 0]
    for t in missed[:5]:
        print("-", t[:150].replace("\n", " "))


if __name__ == "__main__":
    main()
