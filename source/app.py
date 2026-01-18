import streamlit as st
from hybrid_assistant import hybrid_answer
from rag_index import retrieve_context
import pandas as pd

st.set_page_config(page_title="Startel Telecom Assistant", layout="wide")

st.title("📡 Startel Telecom Analytics Assistant")
st.write("Ask questions about revenue, cities, and customer behavior.")

question = st.text_input(
    "Ask a question",
    placeholder="Explain city-wise revenue distribution"
)

if question:
    with st.spinner("Analyzing..."):
        answer = hybrid_answer(question)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("🔍 Retrieved Context (RAG)"):
        st.write(retrieve_context(question))



