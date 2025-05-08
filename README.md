# 🛍️ Sales Prediction Dashboard – Prédisez vos ventes en 3 étapes

> Un dashboard de prévision, des commentaires métier, et une analyse des relations produits à explorer.

---

## 🚀 Essayez l’application

🟢 Application déployée ici :  
👉 **[Streamlit App →](https://sales-prediction-dashboard.streamlit.app/)**  

---

## 🎯 Objectif du projet

Comment anticiper la demande dans le retail ?  
Ce projet simule une problématique concrète : **prévoir les quantités vendues sur plusieurs semaines**, en tenant compte de données réalistes (famille produit, remises, saisonnalité…).

Au-delà de la prévision, il intègre une **lecture structurelle par graphes** :  
- 🔗 **Graphe de co-achats** : pour visualiser les liens entre produits fréquemment achetés ensemble

---

## 🧭 Schéma du pipeline

Voici l’architecture globale du projet, résumée en une image :

![Retail Forecasting Pipeline](Schéma.png)

Ce schéma illustre les **3 grandes étapes** qui composent le cœur du pipeline :

---

### 📥 1. Source de données

- Données simulées à partir de fichiers CSV (`ventes.csv`)
- Historique de ventes multi-produits sur deux ans
- Variables : dates, quantités, remises, prix, familles produits (`hoodie`, `shirt`, `trackwear`)

---

### 🛠️ 2. Préparation des données

- Nettoyage, enrichissement et structuration avec `pandas`
- Typage, gestion des valeurs manquantes, création de variables dérivées (remisé, saison, etc.)

---

### 📊 3. Analyse & Modélisation

Étape centrale du projet, organisée en trois volets :

#### 🔎 Analyse exploratoire
- Bibliothèques : `pandas`, `plotly`
- Objectif : visualiser les ventes par période, famille et produit

#### 🔮 Prédiction
- Modèles : `Prophet`, `XGBoost`
- Objectif : estimer les quantités futures semaine par semaine

#### 🔗 Modélisation graphe
- Outil : `NetworkX`
- Objectif : construire un graphe de co-achats pour explorer les liens entre produits

---

### 🌐 4. Dashboard interactif

- Application construite avec `Streamlit`
- Permet de :
  - Visualiser les prévisions
  - Naviguer dans les résultats par famille de produits
  - Explorer le graphe de co-achats

---

## 🧠 À propos des données

Les données sont **entièrement simulées**, mais structurées à partir de cas réels.  
Elles incluent :
- 🧥 3 familles de produits : `hoodie`, `shirt`, `trackwear`
- 📉 Prix, remises, quantités, dates
- 📆 Un historique de **2 ans** pour capter les effets saisonniers

---

## 👨‍💻 Auteur

Projet développé par **[Lucas Potin](https://lucaspotin98.github.io/)**  
*Data Scientist – Modélisation & Graphes*
