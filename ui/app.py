import streamlit as st
import requests

st.set_page_config(page_title="RAG Router (CPU)", layout="centered")
st.title("RAG Router — CPU-only demo")

backend = st.text_input("API URL", "http://localhost:8008/ask")
q = st.text_area("Your question", "What is RMSE?")

if st.button("Ask"):
    with st.spinner("Requesting..."):
        try:
            r = requests.post(backend, json={"question": q}, timeout=180)
            r.raise_for_status()
            data = r.json()
            st.write(f"Route: **{data['route']}**  |  Timing: **{data['timing_ms']} ms**")
            st.write("### Answer")
            st.write(data["answer"])
            if data.get("passages"):
                st.write("### Retrieved passages")
                for i, p in enumerate(data["passages"], 1):
                    st.markdown(f"**[{i}]** {p['text']}\n\n_Source: {p['source']}_")
        except Exception as e:
            st.error(str(e))
