# HACKATHON IBM 19 - Finance track

Bienvenue sur le dépôt du projet **HACKATHON IBM 19**.  
Ce projet contient : 
1. les certifications des membres de l'équipe obtenues dans le cadre du hackathon --> repertoire "certification"
2. la vidéo de démo requise
3. un dossier de travail avec : 
   - des notebooks Jupyter pour le prétraitement et l'analyse exploratoire des données dans le cadre de l'hackathon.
   - des les données en .csv
   - un notebook avec les différents modèles de prédiction
   - des exports des modèles / visu de watsonx (à confirmer)

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

---

## Structure du projet

1. **Exploration des données (EDA)**  
   - Vérification des valeurs manquantes
   - Formatage des colonnes
   - Statistiques descriptives  
   - Visualisations principales  
   - Obtention d'un dataframe merged avec l'ensemble des données d'entrée 

2. **Analyse avancée et transformations**  
   - Préparation des données  
   - Encodage des variables catégorielles  
   - Analyse des corrélations et PCA  
Cette partie nous a aidé à guider nos choix pour les modèles de prédictions.

3. **Modélisation et conclusions**  
Une partie de la modélisation a été faite sur python dans la continuité du travail précédent mais également sur la plateforme IBM Watsonx - afin de découvrir l'outil et comprendre comment cet outil serait un facilitateur, comparer les performances des différents modèles et surtout voir le gain de temps et d'efficacité (avec le dataframe merged mais également avec celui avant le merge afin de comparer les performances des deux modèle. Ce choix a été fait afin de voir si on pouvait obtenir un modèle quasi autant perfomrant sans les colonnes "en plus" - si c'est le cas, on pourrait utiliser plus facilement le dataframe d'évaluation pour la démonstration que nous avons souhaitée. )

   - Sélection des modèles  
   - Evaluation des performances  


---

