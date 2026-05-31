import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Chat with PDF")
st.title("Chat with PDF")
st.write("Upload a PDF and ask questions about it!")

api_key = st.text_input("Enter your OpenRouter API Key", type="password")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    with st.spinner("Reading PDF..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

    with st.spinner("Splitting into chunks..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(pages)

    with st.spinner("Loading embeddings (first time takes ~1 min)..."):
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    st.success(f"PDF ready! {len(pages)} pages, {len(splits)} chunks.")

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = st.text_input("Ask a question about your PDF")
    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
        st.write("**Answer:**", answer)

else:
    if not api_key:
        st.info("Please enter your OpenRouter API key above.")
    if not uploaded_file:
        st.info("Please upload a PDF file.")