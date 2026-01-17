import pandas as pd
from chromadb import Client
from chromadb.config import Settings
from openai import OpenAI
import os
from pathlib import Path

# ---------- OpenAI ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# ---------- Build documents ----------
def build_documents():
    docs = []

    city_df = pd.read_csv("llm_city_summary.csv")
    for _, row in city_df.iterrows():
        docs.append(
            f"City {row['city']} has {row['total_users']} users, "
            f"generated total revenue {row['total_revenue']}, "
            f"with an average bill of {row['avg_bill']}."
        )

    cust_df = pd.read_csv("llm_customer_revenue.csv")
    for _, row in cust_df.iterrows():
        docs.append(
            f"Customer with ID {row['customer_id']} paid a total of {row['total_paid']}, "
            f"had an average monthly bill of {row['avg_monthly_bill']}, "
            f"was active for {row['active_months']} months, "
            f"with a max bill of {row['max_bill']} "
            f"and min bill of {row['min_bill']}."
        )

    return docs


# ---------- Embeddings ----------
def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in res.data]


# ---------- Build vector DB ----------
def build_vector_db():
    chroma = Client(
        Settings(
            persist_directory=str(CHROMA_DIR),
            is_persistent=True,          # ✅ THIS IS THE KEY
            anonymized_telemetry=False
        )
    )

    collection = chroma.get_or_create_collection("startel_rag")

    docs = build_documents()
    embeddings = embed(docs)

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(docs))]
    )

    print("✅ Documents added:", collection.count())
    print("📁 Persist directory:", CHROMA_DIR)




# ---------- Retrieve context ----------
def retrieve_context(query, k=5):
    chroma = Client(
        Settings(
            persist_directory=str(CHROMA_DIR),
            is_persistent=True,
            anonymized_telemetry=False
        )
    )

    collection = chroma.get_or_create_collection("startel_rag")

    query_embedding = embed([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant context found."

    return "\n".join(results["documents"][0])


