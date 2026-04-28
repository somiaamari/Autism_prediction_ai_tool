import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class AutismPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.accuracy = None
        self.label_encoders = {}  # Pour les encodeurs si nécessaire
        self.load_model()
    
    def load_model(self):
        """Charge le modèle entraîné"""
        try:
            model_path = Path("models/autism_model.pkl")
            
            if not model_path.exists():
                print("⚠️  Modèle non trouvé. Lancez l'entraînement d'abord.")
                return False
            
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.accuracy = model_data['accuracy']
            
            print(f"✅ Modèle chargé (Précision: {self.accuracy:.2%})")
            print(f"📊 Features attendues: {len(self.feature_names)}")
            print(f"🔍 5 premières features: {self.feature_names[:5]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False
    
    def prepare_features(self, form_data):
        """
        Prépare les données du formulaire pour la prédiction
        form_data: dictionnaire avec les clés:
          - age: int
          - gender: 'Male'/'Female'
          - jundice: 'Yes'/'No'
          - family_asd: 'Yes'/'No'
          - A1_Score à A10_Score: 0 ou 1  # <-- NOTE: A1_Score not A1
        """
        
        print("\n🔍 DEBUG - Données reçues:")
        for key, value in sorted(form_data.items()):
            print(f"  {key}: {value}")
        
        # 1. Initialiser toutes les features à 0
        features = {feature: 0 for feature in self.feature_names}
        
        # 2. Remplir les scores A1-A10 - CORRECTED VERSION
        print("\n🔍 DEBUG - Traitement des scores A1-A10:")
        for i in range(1, 11):
            # Get value from form_data - FIXED: check A{i}_Score first
            value = 0
            key_with_score = f'A{i}_Score'
            key_simple = f'A{i}'
            
            if key_with_score in form_data:
                value = form_data[key_with_score]
                print(f"  ✓ Found {key_with_score} = {value}")
            elif key_simple in form_data:
                value = form_data[key_simple]
                print(f"  ✓ Found {key_simple} = {value} (fallback)")
            else:
                print(f"  ⚠️  No key found for A{i}, using 0")
            
            # Trouver le bon nom de feature dans le modèle
            feature_found = False
            possible_names = [
                f"A{i}_Score",    # Format le plus probable
                f"A{i}",          # Format simple
                f"a{i}_score",    # Minuscule
                f"Q{i}"           # Autre format
            ]
            
            for name in possible_names:
                if name in self.feature_names:
                    features[name] = value
                    print(f"    → Set {name} = {value}")
                    feature_found = True
                    break
            
            if not feature_found:
                print(f"    ⚠️  No matching feature found for A{i}")
        
        # 3. Traiter l'âge
        print(f"\n🔍 DEBUG - Traitement de l'âge:")
        age = form_data.get('age', 5)
        print(f"  ✓ Âge brut: {age}")
        
        # Normaliser l'âge
        try:
            # IMPORTANT: Le scaler attend un DataFrame 2D
            age_array = np.array([[float(age)]])
            age_normalized = self.scaler.transform(age_array)[0][0]
            features['age'] = age_normalized
            print(f"  ✓ Âge normalisé: {age_normalized:.4f}")
        except Exception as e:
            print(f"  ⚠️  Erreur normalisation âge: {e}")
            features['age'] = 0.5  # Valeur par défaut
        
        # 4. Variables catégorielles (encodage one-hot)
        print(f"\n🔍 DEBUG - Traitement des variables catégorielles:")
        gender = form_data.get('gender', 'Male')
        jundice = form_data.get('jundice', 'No')
        family_asd = form_data.get('family_asd', 'No')
        
        print(f"  ✓ Genre: {gender}")
        print(f"  ✓ Jaundice: {jundice}")
        print(f"  ✓ Antécédents familiaux: {family_asd}")
        
        # Chercher les noms exacts des colonnes one-hot
        gender_col = None
        jundice_col = None
        family_col = None
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            if 'gender' in feature_lower:
                gender_col = feature
            elif 'jundice' in feature_lower:
                jundice_col = feature
            elif 'family' in feature_lower or 'asd' in feature_lower:
                family_col = feature
        
        # Mapper les valeurs
        if gender_col:
            # Si la colonne contient '_m' ou 'male', c'est pour Male=1
            if '_m' in gender_col.lower() or 'male' in gender_col.lower():
                features[gender_col] = 1 if gender == 'Male' else 0
            else:
                # Sinon, on suppose que c'est pour Female=1
                features[gender_col] = 1 if gender == 'Female' else 0
            print(f"  ✓ {gender_col} = {features[gender_col]}")
        
        if jundice_col:
            # Si la colonne contient '_yes', c'est pour Yes=1
            if '_yes' in jundice_col.lower():
                features[jundice_col] = 1 if jundice == 'Yes' else 0
            else:
                # Sinon, on suppose que c'est pour No=1
                features[jundice_col] = 1 if jundice == 'No' else 0
            print(f"  ✓ {jundice_col} = {features[jundice_col]}")
        
        if family_col:
            # Même logique pour les antécédents familiaux
            if '_yes' in family_col.lower():
                features[family_col] = 1 if family_asd == 'Yes' else 0
            else:
                features[family_col] = 1 if family_asd == 'No' else 0
            print(f"  ✓ {family_col} = {features[family_col]}")
        
        # 5. Créer le DataFrame dans le bon ordre
        df = pd.DataFrame([features])
        
        # Réorganiser pour avoir les colonnes dans le bon ordre
        df = df[self.feature_names]
        
        print(f"\n🔍 DEBUG - DataFrame final:")
        print(f"  Shape: {df.shape}")
        print(f"  Colonnes: {list(df.columns)}")
        
        # Afficher les valeurs A1-A10 pour débogage
        print(f"  Valeurs A1-A10 stockées:")
        for i in range(1, 11):
            feature_name = f"A{i}_Score"
            if feature_name in df.columns:
                val = df[feature_name].iloc[0]
                print(f"    {feature_name}: {val}")
        
        print(f"  Valeurs non nulles (A1-A10):")
        non_zero_count = 0
        for i in range(1, 11):
            feature_name = f"A{i}_Score"
            if feature_name in df.columns:
                val = df[feature_name].iloc[0]
                if val != 0:
                    print(f"    {feature_name}: {val}")
                    non_zero_count += 1
        
        print(f"  Total A1-A10 non nul: {non_zero_count}/10")
        
        return df
    
    def _get_recommendation(self, probability):
        """Get recommendation based on probability"""
        if probability <= 0.3:
            return "Continue with regular developmental monitoring."
        elif probability <= 0.6:
            return "Consider discussing these results with your pediatrician."
        elif probability <= 0.85:
            return "Professional developmental evaluation is recommended."
        else:
            return "Urgent evaluation with a specialist is recommended."
    
    def predict(self, form_data):
        """
        Prédit la probabilité d'autisme
        
        Returns:
            float: probabilité entre 0 et 1
            str: niveau de risque
            dict: détails de la prédiction
        """
        try:
            if not self.model:
                print("❌ Modèle non chargé")
                return 0.5, "🟡 Estimation", {}
            
            # Préparer les features
            X = self.prepare_features(form_data)
            
            # Vérifier que le DataFrame n'est pas vide
            if X.empty:
                print("❌ DataFrame vide après préparation")
                # Fallback: calcul basé sur les réponses
                a_scores = []
                for i in range(1, 11):
                    if f'A{i}_Score' in form_data:
                        a_scores.append(form_data[f'A{i}_Score'])
                    elif f'A{i}' in form_data:
                        a_scores.append(form_data[f'A{i}'])
                    else:
                        a_scores.append(0)
                
                fallback_prob = sum(a_scores) / 10 if a_scores else 0.5
                
                # Déterminer le niveau de risque
                if fallback_prob <= 0.3:
                    risk = "🟢 Low Risk"
                elif fallback_prob <= 0.6:
                    risk = "🟡 Moderate Risk"
                elif fallback_prob <= 0.85:
                    risk = "🟠 High Risk"
                else:
                    risk = "🔴 Very High Risk"
                
                return fallback_prob, risk, {'fallback': True}
            
            # Faire la prédiction
            print(f"\n🎯 Prédiction en cours...")
            probability = self.model.predict_proba(X)[0][1]
            
            # Déterminer le niveau de risque (4 niveaux)
            if probability <= 0.3:  # 0-30%
                risk_level = "🟢 Low Risk"
            elif probability <= 0.6:  # 31-60%
                risk_level = "🟡 Moderate Risk"
            elif probability <= 0.85:  # 61-85%
                risk_level = "🟠 High Risk"
            else:  # 86-100%
                risk_level = "🔴 Very High Risk"
            
            # Calculer un score basé sur les réponses A1-A10
            a_scores = []
            for i in range(1, 11):
                if f'A{i}_Score' in form_data:
                    a_scores.append(form_data[f'A{i}_Score'])
                elif f'A{i}' in form_data:
                    a_scores.append(form_data[f'A{i}'])
                else:
                    a_scores.append(0)
            
            concern_count = sum(a_scores)
            alternative_score = sum(a_scores) / 10 if a_scores else 0
            
            details = {
                'probability': float(probability),
                'risk_level': risk_level,
                'concern_count': concern_count,
                'total_questions': 10,
                'model_accuracy': self.accuracy,
                'features_used': len(self.feature_names),
                'recommendation': self._get_recommendation(probability),
                'alternative_score': alternative_score
            }
            
            print(f"✅ Prédiction terminée: {probability:.2%} ({risk_level})")
            print(f"   Concern count: {concern_count}/10")
            print(f"   Alternative score: {alternative_score:.1%}")
            
            return probability, risk_level, details
            
        except Exception as e:
            print(f"❌ Erreur de prédiction: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: calcul basé sur les réponses A1-A10
            a_scores = []
            for i in range(1, 11):
                if f'A{i}_Score' in form_data:
                    a_scores.append(form_data[f'A{i}_Score'])
                elif f'A{i}' in form_data:
                    a_scores.append(form_data[f'A{i}'])
                else:
                    a_scores.append(0)
            
            fallback_prob = sum(a_scores) / 10 if a_scores else 0.5
            
            # Ajuster avec les facteurs de risque
            if form_data.get('family_asd') == 'Yes':
                fallback_prob = min(1, fallback_prob + 0.2)
            if form_data.get('jundice') == 'Yes':
                fallback_prob = min(1, fallback_prob + 0.1)
            
            # Déterminer le niveau de risque
            if fallback_prob <= 0.3:
                risk = "🟢 Low Risk"
            elif fallback_prob <= 0.6:
                risk = "🟡 Moderate Risk"
            elif fallback_prob <= 0.85:
                risk = "🟠 High Risk"
            else:
                risk = "🔴 Very High Risk"
            
            return fallback_prob, risk, {
                'fallback': True,
                'concern_count': sum(a_scores),
                'method': 'fallback_simple',
                'recommendation': self._get_recommendation(fallback_prob)
            }
    
    def predict_for_streamlit(self, form_data):
        """
        Enhanced prediction method for Streamlit display
        Uses the same logic as predict() but ensures consistency
        """
        return self.predict(form_data)
    
    def test_form_data_reading(self):
        """Test if the predictor can read form data correctly"""
        test_data = {
            'A1_Score': 1,
            'A2_Score': 1,
            'A3_Score': 1,
            'A4_Score': 1,
            'A5_Score': 1,
            'A6_Score': 1,
            'A7_Score': 1,
            'A8_Score': 1,
            'A9_Score': 1,
            'A10_Score': 1,
            'age': 5,
            'gender': 'Male',
            'jundice': 'Yes',
            'family_asd': 'No'
        }
        
        print("\n🧪 TESTING FORM DATA READING")
        print("="*50)
        
        X = self.prepare_features(test_data)
        
        # Check if A1-A10 values are 1
        for i in range(1, 11):
            col_name = f"A{i}_Score"
            if col_name in X.columns:
                value = X[col_name].iloc[0]
                print(f"{col_name}: {value} {'✅' if value == 1 else '❌'}")
            else:
                print(f"{col_name}: NOT FOUND ❌")
        
        print("="*50)
        return X

# Instance globale pour l'application
predictor = AutismPredictor()

def predict_probability(form_data):
    """Fonction simple pour Streamlit - retourne juste la probabilité"""
    probability, _, _ = predictor.predict(form_data)
    return probability


# Fonction de test direct
def test_predictor():
    """Teste le prédicteur avec des données de test"""
    print("\n" + "="*50)
    print("🧪 TEST DIRECT DU PRÉDICTEUR")
    print("="*50)
    
    # Cas de test - USING A{i}_Score NOT A{i}
    test_cases = [
        ("Cas faible risque", {
            'age': 4,
            'gender': 'Male',
            'jundice': 'No',
            'family_asd': 'No',
            'A1_Score': 0, 'A2_Score': 0, 'A3_Score': 0, 'A4_Score': 0, 'A5_Score': 0,
            'A6_Score': 0, 'A7_Score': 0, 'A8_Score': 0, 'A9_Score': 0, 'A10_Score': 0
        }),
        ("Cas moyen risque", {
            'age': 3,
            'gender': 'Female',
            'jundice': 'No',
            'family_asd': 'Yes',
            'A1_Score': 1, 'A2_Score': 1, 'A3_Score': 0, 'A4_Score': 1, 'A5_Score': 1,
            'A6_Score': 0, 'A7_Score': 1, 'A8_Score': 0, 'A9_Score': 0, 'A10_Score': 0
        }),
        ("Cas haut risque", {
            'age': 5,
            'gender': 'Male',
            'jundice': 'Yes',
            'family_asd': 'Yes',
            'A1_Score': 1, 'A2_Score': 1, 'A3_Score': 1, 'A4_Score': 1, 'A5_Score': 1,
            'A6_Score': 1, 'A7_Score': 1, 'A8_Score': 1, 'A9_Score': 1, 'A10_Score': 1
        })
    ]
    
    for name, data in test_cases:
        print(f"\n📋 {name}:")
        prob, risk, details = predictor.predict(data)
        print(f"  📈 Probabilité: {prob:.2%}")
        print(f"  ⚠️  Risque: {risk}")
        print(f"  🔍 Détails: concern_count={details.get('concern_count')}, alternative_score={details.get('alternative_score', 0):.1%}")
    
    print("\n" + "="*50)
    print("✅ TEST COMPLÉTÉ")
    print("="*50)


if __name__ == "__main__":
    # Test form data reading first
    print("\n" + "="*50)
    print("🧪 PREMIER TEST: LECTURE DES DONNÉES")
    print("="*50)
    predictor.test_form_data_reading()
    
    # Then run the full test
    test_predictor()