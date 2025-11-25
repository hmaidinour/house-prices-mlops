import pandas as pd
import os

# Définition de la fonction comme demandé [cite: 627]
def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    print(f"Chargement des données depuis {input_path}...")
    df = pd.read_csv(input_path)
    
    # Sélection des features numériques seulement [cite: 74]
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    
    # Suppression des lignes avec valeurs manquantes [cite: 74]
    df_clean = df_numeric.dropna()
    
    # Création du dossier de sortie si inexistant
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde des données train et test prêtes
    output_path = os.path.join(output_dir, "train_cleaned.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"Données nettoyées sauvegardées dans {output_path}")

# Appel de la fonction [cite: 629]
if __name__ == "__main__":
    prepare_data()