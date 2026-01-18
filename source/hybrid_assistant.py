from analytics import load_data, answer_analytical
from rag_index import retrieve_context
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def detect_intent(question):
    q = question.lower()

    analytical_patterns = [
        "total revenue",
        "highest revenue",
        "which year",
        "how many",
        "top customer",
        "highest contributor",
        "list",
        "what all",
        "all cities",
        "show cities"
    ]

    if any(p in q for p in analytical_patterns):
        return "ANALYTICAL"

    return "RAG"

def detect_intent(question):
    q = question.lower()

    analytical_patterns = [
        "total revenue in",
        "highest revenue",
        "max revenue",
        "which year",
        "how many",
        "top customer",
        "highest contributor"
    ]

    if any(p in q for p in analytical_patterns):
        return "ANALYTICAL"

    return "RAG"



def answer_rag(question):
    context = retrieve_context(question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a telecom analytics assistant. "
                    "Answer using ONLY the given context. "
                    "If the answer is not in the context, say so."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )

    return response.choices[0].message.content


def hybrid_answer(question):
    events_df, city_df, customer_df = load_data()
    intent = detect_intent(question)

    if intent == "ANALYTICAL":
        return answer_analytical(question, events_df, city_df, customer_df)

    return answer_rag(question)
