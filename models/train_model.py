import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def train_autism_model():
    """
    Entraîne le modèle Naive Bayes pour la détection d'autisme
    Retourne le modèle entraîné et les métriques
    """
    print("🎯 Chargement des données...")
    
    # 1. Charge les datasets (comme dans ton notebook)
    df1 = pd.read_csv("data/Child-Data2017.csv")
    df2 = pd.read_csv("data/Child-Data2018.csv")
    
    # 2. Nettoyage (identique à ton notebook)
    df2.columns = df2.columns.str.strip()
    df2_clean = df2.rename(columns={
        "A1": "A1_Score", "A2": "A2_Score", "A3": "A3_Score",
        "A4": "A4_Score", "A5": "A5_Score", "A6": "A6_Score",
        "A7": "A7_Score", "A8": "A8_Score", "A9": "A9_Score",
        "A10": "A10_Score", "Age": "age", "Sex": "gender",
        "Ethnicity": "ethnicity", "Jaundice": "jundice",
        "Family_ASD": "austim", "Residence": "contry_of_res",
        "Used_App_Before": "used_app_before", "Score": "result",
        "Screening Type": "age_desc", "User": "relation",
        "Class": "Class/ASD"
    })
    
    # 3. Concaténation
    df_final = pd.concat([df1, df2_clean], ignore_index=True, sort=False)
    df_final.rename(columns={"austim": "Family_ASD"}, inplace=True)
    
    # 4. Suppression des colonnes inutiles
    df_final = df_final.drop(
        ["id", "ethnicity", "contry_of_res", "used_app_before", 
         "result", "age_desc", "relation", "Case No", 
         "Why taken the screening", "Language"],
        axis=1,
        errors='ignore'
    )
    
    # 5. Nettoyage de l'âge
    df_final = df_final[df_final["age"] != "?"]
    df_final['age'] = pd.to_numeric(df_final['age'], errors='coerce')
    
    # 6. Gestion des valeurs manquantes pour la cible
    df_final['Class/ASD'] = df_final['Class/ASD'].fillna("Unknow").astype(str)
    
    # 7. Encodage des variables catégorielles
    categorical_cols = ['gender', 'jundice', 'Family_ASD']
    df_final = pd.get_dummies(df_final, columns=categorical_cols, drop_first=True)
    
    # 8. Normalisation de l'âge
    scaler = MinMaxScaler()
    df_final['age'] = scaler.fit_transform(df_final[['age']])
    
    # 9. Encodage de la cible
    df_final['Class/ASD'] = df_final['Class/ASD'].map({
        'YES': 1,
        'NO': 0,
        'Unknow': -1
    })
    
    # 10. Séparation données étiquetées/non-étiquetées
    df_labeled = df_final[df_final['Class/ASD'] != -1]
    df_unlabeled = df_final[df_final['Class/ASD'] == -1]
    
    print(f"📊 Données étiquetées: {len(df_labeled)}")
    print(f"📊 Données non étiquetées: {len(df_unlabeled)}")
    
    # 11. Préparation pour l'entraînement
    X = df_labeled.drop('Class/ASD', axis=1)
    y = df_labeled['Class/ASD']
    
    # 12. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=104, train_size=0.8, shuffle=True
    )
    
    print(f"🎯 Entraînement sur {len(X_train)} échantillons...")
    
    # 13. Entraînement du modèle Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # 14. Évaluation
    y_pred = gnb.predict(X_test)
    y_pred_proba = gnb.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Précision: {accuracy:.2%}")
    print(f"✅ AUC Score: {roc_auc:.2%}")
    
    # 15. Sauvegarde du modèle
    model_data = {
        'model': gnb,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'train_shape': X_train.shape
    }
    
    with open("models/autism_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("💾 Modèle sauvegardé dans: models/autism_model.pkl")
    
    # Sauvegarde aussi les noms des colonnes pour référence
    column_info = {
        'features': list(X.columns),
        'categorical_cols': categorical_cols
    }
    
    with open("models/column_info.pkl", "wb") as f:
        pickle.dump(column_info, f)
    
    return gnb, accuracy, roc_auc

if __name__ == "__main__":
    train_autism_model()