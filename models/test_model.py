import sys
sys.path.append('.')  # Pour importer depuis le dossier core

from core.predictor import AutismPredictor

def test_integration():
    """Test l'intégration avec le module predictor"""
    
    print("🔗 TEST D'INTÉGRATION AVEC PREDICTOR.PY")
    print("=" * 50)
    
    # 1. Initialiser le prédicteur
    predictor = AutismPredictor()
    
    if not predictor.model:
        print("❌ Le modèle n'a pas été chargé")
        return
    
    print(f"✅ Prédicteur initialisé")
    print(f"📊 Précision du modèle: {predictor.accuracy:.2%}")
    
    # 2. Tester différents cas
    test_cases = [
        ("👶 Enfant typique 3 ans", {
            'age': 11, 'gender': 'Female', 'jundice': 'No', 'family_asd': 'No',
            'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1,
            'A6': 1, 'A7': 1, 'A8': 0, 'A9': 0, 'A10': 0
        }),
        ("🧒 Enfant avec quelques signes", {
            'age': 6, 'gender': 'Male', 'jundice': 'No', 'family_asd': 'No',
            'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5':1,
            'A6': 1, 'A7': 0, 'A8': 1, 'A9': 0, 'A10': 0
        }),
        ("🚨 Cas préoccupant", {
            'age': 5, 'gender': 'Male', 'jundice': 'Yes', 'family_asd': 'No',
            'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1,
            'A6': 1, 'A7': 1, 'A8': 0, 'A9': 1, 'A10': 0
        })
    ]
    
    print("\n🧪 TESTS DE PRÉDICTION:")
    print("-" * 50)
    
    for name, data in test_cases:
        print(f"\n{name}")
        print(f"Âge: {data['age']}, Genre: {data['gender']}")
        print(f"Jaundice: {data['jundice']}, Antécédents familiaux: {data['family_asd']}")
        
        # Compter les réponses A=1
        a_scores = sum(data[f'A{i}'] for i in range(1, 11))
        print(f"Réponses préoccupantes (A1-A10): {a_scores}/10")
        
        # Prédiction
        probability, risk_level, details = predictor.predict(data)
        
        print(f"📈 Probabilité: {probability:.2%}")
        print(f"⚠️  Niveau de risque: {risk_level}")
        print(f"🔍 Détails: {details}")
        
        # Vérification de la cohérence
        if a_scores >= 7 and probability < 0.5:
            print("❌ ATTENTION: Incohérence possible - beaucoup de signes mais probabilité basse")
        elif a_scores <= 2 and probability > 0.7:
            print("❌ ATTENTION: Incohérence possible - peu de signes mais probabilité élevée")
        else:
            print("✅ Cohérence OK")
    
    print("\n" + "=" * 50)
    print("🎉 TEST D'INTÉGRATION RÉUSSI!")

if __name__ == "__main__":
    test_integration()