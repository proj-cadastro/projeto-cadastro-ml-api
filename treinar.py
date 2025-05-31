from src.data.load_data import load_professores
from src.ml.preprocessing import preprocess_professores
from src.ml.train_models import train_professor_models

def main():
    df = load_professores()
    modelos_dir = "modelos_treinados"
    processed = preprocess_professores(df, save_dir=modelos_dir)
    train_professor_models(processed, save_dir=modelos_dir)
    print(f"Modelos treinados e salvos em '{modelos_dir}'.")

if __name__ == "__main__":
    main()
