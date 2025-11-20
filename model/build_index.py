import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

#
df = pd.read_csv("qa_uniform.csv")
df = df.dropna(subset=["response", "context"])

# Texte qui servira de base de connaissance (tu peux adapter)
df["kb_text"] = df["question"] + " " + df["context"]

texts = df["kb_text"].tolist()

# Modèle d'embeddings (open-source, pas un LLM complet)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encoder tous les textes en vecteurs
embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # important pour la similarité cosinus
)

embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

d = embeddings.shape[1]  # dimension des vecteurs

# Construire l'index FAISS (Inner Product ≈ cosinus car normalisé)
index = faiss.IndexFlatIP(d)
index.add(embeddings) # type: ignore[arg-type]

print("Nb de vecteurs dans l'index :", index.ntotal)

# Sauvegarder l'index et le dataframe minimal
faiss.write_index(index, "kb.index")
df[["question", "context", "response"]].to_parquet("kb_rows.parquet")

# Sauvegarder aussi le nom du modèle d'embeddings pour réutilisation
with open("embed_model_name.pkl", "wb") as f:
    pickle.dump("all-MiniLM-L6-v2", f)

print("Index FAISS et données sauvegardés.")
