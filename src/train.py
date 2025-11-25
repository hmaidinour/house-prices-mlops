import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Définition de la fonction [cite: 637]
def train_model(data_dir="data/processed", output_dir="models"):
    print("Chargement des données pour l'entraînement...")
    data_path = os.path.join(data_dir, "train_cleaned.csv")
    df = pd.read_csv(data_path)
    
    # Séparation X (Features) et y (Target: SalePrice) [cite: 75]
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    
    # Split train/test [cite: 77]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement [cite: 79]
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Modèle entraîné.")
    
    # Sauvegarde du modèle [cite: 80]
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}")
    
    # On peut sauvegarder X_test et y_test pour l'évaluation plus tard
    X_test.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)

if __name__ == "__main__":
    train_model()