# PromptShield AI

Pare-feu pour chatbot LLM : détecte les injections de prompts et décide d'**autoriser / alerter / refuser** un message avant qu'il n'atteigne le chatbot.

Projet 04 du contrôle continu "IA pour la Cybersécurité" (EMSI).

## Équipe
- Étudiant(e) 1 : ...
- Étudiant(e) 2 : ...

## Installation

```bash
python -m venv venv
# Windows : venv\Scripts\activate
# Linux/Mac : source venv/bin/activate
pip install -r requirements.txt
```

## Lancer la baseline (règles)

```bash
python baseline_rules.py
```

## Données
- Source : https://huggingface.co/datasets/deepset/prompt-injections (662 entrées, colonnes `text` et `label`, 1 = injection)
- Partition : train/test officiels du dataset. Le jeu de test n'est jamais utilisé pour l'entraînement.
- Jeu de tests indépendant (40 à 60 cas, injections directes et indirectes) : à créer dans `data/custom_tests.csv`.

## Approches comparées
1. Baseline : règles
2. ML : TF-IDF + SVM
3. (Facultatif) Transformer pré-entraîné
4. Chatbot cible : chatbot simulé ou LLM local (Ollama)

## Avancement
- [x] Structure du projet
- [ ] Baseline par règles évaluée
- [ ] Jeu de tests indépendant
- [ ] TF-IDF + SVM
- [ ] API de garde (FastAPI) + seuils + journal
- [ ] Comparaison avant/après protection
- [ ] Rapport et soutenance

## Limites
Aucune attaque sur un service réel : tout reste en laboratoire, sur un chatbot fictif.
