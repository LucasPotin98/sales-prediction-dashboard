# 🛍️ 12 semaines pour prédire la demande — une app Streamlit pour le retail

> 🧠 3 modèles. 🛍️ 3 familles. 🔁 1 pipeline glissant pour anticiper les ventes.

---

## 🎯 Pourquoi ce projet ?

Comment prévoir la demande de vêtements à moyen terme ?  
Ce projet simule une problématique concrète du retail : **prédire les quantités vendues sur plusieurs semaines**, à partir de données réalistes (produits, remises, saisonnalité…).


---

## 🧭 Architecture du projet

![Retail Forecasting Pipeline](assets/schema_forecasting.png)

---

## 🚀 Essayez l’application

🟢 Application déployée ici :  
**[Streamlit App →](https://tonapp.streamlit.app/)**

Aucune installation nécessaire.

---

## 📦 Ce que contient ce projet

| Élément | Description |
|--------|-------------|
| 🎛️ App Streamlit | Navigation multipage (Contexte, Analyse, Modélisation, Graphe) |
| 🧠 Modèles | Naïf (baseline), XGBoost, Prophet |
| 🔁 Pipeline prédictif | Prédiction glissante semaine par semaine |
| 📅 Choix dynamique | Sélection de la famille produit + horizon de prévision |
| 📈 Résultats | Graphe réel vs prédit + métriques (RMSE, MAE, R²) |

---

## 🧪 Exemple : Prédiction pour la famille “Hoodie” sur 12 semaines

![Screenshot](assets/screenshot_app.png)

> Le modèle XGBoost capture bien la saisonnalité, tandis que Prophet anticipe mieux les pics.

---

## ⚙️ Stack technique

| Étape | Outils |
|-------|--------|
| 🧹 Préparation | Pandas, NumPy |
| 🔧 Feature Engineering | Rolling, Lags, Variation, Temporal |
| 🧠 Modélisation | XGBoost, Prophet, modèle naïf custom |
| 📊 Interface | Streamlit |
| 📦 Stockage modèles | Joblib / JSON |
| 🛠 Déploiement | Streamlit Cloud |

---

## 🧠 Un mot sur la donnée

Les données sont **simulées** à partir de schémas de vente inspirés du réel.  
Elles contiennent :
- 3 familles de produits (`hoodie`, `shirt`, `trackwear`)
- des remises, des dates, des prix, des quantités
- un historique sur **plus d’un an**, pour capter les effets saisonniers


---

## ✍️ Auteur

Projet développé par [Lucas Potin](https://lucaspotin98.github.io/.fr)  
Data Scientist | Modélisation & Graphes