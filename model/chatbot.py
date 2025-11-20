import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Recharger l'index et les données
index = faiss.read_index("kb.index")
df = pd.read_parquet("kb_rows.parquet")

# Recharger le modèle d'embeddings
with open("embed_model_name.pkl", "rb") as f:
    model_name = pickle.load(f)

embed_model = SentenceTransformer(model_name)


def retrieve(question: str, k: int = 5) -> pd.DataFrame:
    """Retourne les k lignes les plus proches pour une question."""
    q_emb = embed_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    D, I = index.search(q_emb, k)  # I = indices des lignes
    return df.iloc[I[0]]           # DataFrame de k lignes


def answer_question(question: str, k: int = 5) -> str:
    """Répond à la question en renvoyant la 'response' de la meilleure ligne."""
    retrieved = retrieve(question, k=k)

    # la meilleure ligne = première
    best = retrieved.iloc[0]

    # Optionnel : tu peux aussi utiliser best["context"] si tu veux afficher le contexte
    return best["response"]


#########################
print(answer_question("Who is the 2009 Dana Vikings Head Coach?"))