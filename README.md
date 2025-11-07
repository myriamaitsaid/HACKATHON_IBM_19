# HACKATHON IBM 19 - Finance track

Bienvenue sur le dépôt du projet **HACKATHON IBM 19**.  
Ce projet contient : 
1. les certifications des membres de l'équipe obtenues dans le cadre du hackathon --> repertoire "certification"
2. la vidéo de démo requise
3. un dossier de travail avec : 
   - des notebooks Jupyter pour le prétraitement et l'analyse exploratoire des données dans le cadre de l'hackathon.
   - les données en .csv
   - un notebook avec les différents modèles de prédiction

---

## Contenu du dossier de travail

| Fichier | Description |
|--------|-------------|
| `EDA_p1.ipynb` | Prétraitement et EDA des données (users_data, cards_data, mcc_codes, labels) |
| `EDA_p2.ipynb` | Prétraitement et EDA des données (transaction et features) |
| `EDA_p3.ipynb` | Merge |
| `FeatureEngineeringIBM.ipynb` | EDA, visualisation |
| `prediction.ipynb` | construction des modèles de prédiction|
| `data/raw` | dossier pour les fichiers de données initiaux|
| `data/clean` | dossier pour les fichiers de données propres|
| `fraud_app` | dossier pour l'application streamlit|


---

## Structure du projet

### Exploration des données (EDA)
- **Vérification des valeurs manquantes** : identification des colonnes contenant des données manquantes et traitement approprié.  
- **Formatage des colonnes** : uniformisation des types et noms des colonnes pour faciliter les analyses.  
- **Statistiques descriptives** : calcul des principales mesures (moyenne, médiane, écart-type, etc.) pour mieux comprendre la distribution des variables.  
- **Visualisations principales** : graphiques et représentations visuelles permettant d’identifier tendances, anomalies et relations entre variables.  
- **Obtention d’un dataframe merged** : fusion de l’ensemble des données d’entrée pour disposer d’un jeu de données complet et cohérent.

### Analyse avancée et transformations
- **Préparation des données** : nettoyage final et organisation des données pour la modélisation.  
- **Encodage des variables catégorielles** : transformation des colonnes catégorielles en variables numériques exploitables par les modèles.  
- **Analyse des corrélations et PCA** : étude des relations entre variables et réduction de dimension pour guider les choix des modèles de prédiction.

### Modélisation et conclusions
- Une partie de la modélisation a été réalisée en **Python**, dans la continuité de l’EDA, et également sur la **plateforme IBM Watsonx**.  
- L’objectif était de découvrir l’outil, comprendre comment il peut faciliter le workflow, comparer les performances des modèles et évaluer le gain de temps et d’efficacité.  
- Deux jeux de données ont été utilisés :  
  - **Dataframe merged** : avec toutes les colonnes disponibles.  
  - **Dataframe avant fusion** : afin de tester si un modèle performant peut être obtenu sans certaines colonnes supplémentaires.  
- Cette démarche permet de déterminer si un modèle quasi aussi performant peut être utilisé avec un jeu de données plus simple pour la démonstration.

### Sélection des modèles
- Identification et choix des modèles les plus pertinents pour la prédiction.
- Sur Watsonx, nous avons testé tous les modèles de classification disponibles en sélectionnant la métrique de recall comme critère principal. Cela inclut notamment la régression linéaire, la régression logistique, les forêts aléatoires, et les autres modèles proposés automatiquement par la plateforme.
- En parallèle, sur Python en local, nous avons évalué plusieurs modèles :
   - SVM (avec SMOTE, Borderline-SMOTE et sans SMOTE),
   - Random Forest (avec SMOTE, Borderline-SMOTE et sans SMOTE),
   - AdaBoost (avec et sans PCA, avec et sans SMOTE).
   - XGBoost (avec et sans PCA, avec et sans SMOTE).
 

### Évaluation des performances
- Mesure et comparaison des performances des modèles sélectionnés afin de déterminer le meilleur candidat pour la mise en production ou la démonstration.

Voici le lien de la vidéo MVP : https://drive.google.com/drive/folders/1Jd0pp3bnq1gFS0Snbb9IR-eOpDVGqXaa?usp=drive_link

