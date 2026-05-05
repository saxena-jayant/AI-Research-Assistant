import streamlit as st
from agents.graph import build_graph

st.title("🔍 Multi-Agent Research Assistant")

query = st.text_input("Enter your query:")

if st.button("Research"):
    if query:
        with st.spinner("Thinking..."):
            graph = build_graph()
            inputs = {"query": query, "retry_count": 0, "refined_query": ""}
            outputs = graph.invoke(inputs)
            st.write("### Final summary:")
            st.success(outputs["summary"])
    else:
        st.warning("Please enter a query.")
