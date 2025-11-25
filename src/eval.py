import pandas as pd
import joblib
import os
# On importe la nouvelle fonction root_mean_squared_error
from sklearn.metrics import root_mean_squared_error, r2_score

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):
    print("Chargement du modèle et des données de test...")
    model = joblib.load(model_path)
    
    # Chargement des données de test sauvegardées lors de l'entraînement
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
    
    # Prédiction
    predictions = model.predict(X_test)
    
    # Calcul des métriques (Correction ici : on utilise la fonction directe)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"--- Résultat de l'évaluation ---")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

if __name__ == "__main__":
    evaluate_model()