import joblib
from sklearn.ensemble import GradientBoostingClassifier

# 1. Carregar o modelo usando joblib (em vez de pickle)
try:
    modelo = joblib.load('bearing_predictor_gb.pkl')
    print("✅ Modelo carregado com sucesso!")
    
    # 2. Visualizar os parâmetros principais
    print("\n--- Parâmetros do Modelo ---")
    print(modelo.get_params())
    
    # 3. Ver se o modelo já foi treinado (atributos que terminam em _)
    if hasattr(modelo, "n_estimators_"):
        print(f"\nO modelo foi treinado com {modelo.n_estimators_} estimadores.")

except Exception as e:
    print(f"❌ Erro ao carregar: {e}")