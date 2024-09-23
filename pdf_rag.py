from langchain_community.document_loaders import PyPDFLoader
import getpass
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.schema import Document
import markdown
import streamlit as st

openai = st.secrets.db_credentials.openai
nvidia = st.secrets.db_credentials.nvidia
langchain = st.secrets.db_credentials.langchain

llm = ChatNVIDIA(base_url="http://localhost:8000/v1",model="meta/llama3-70b-instruct")

def pdf_rag(file_path, user_input):
    #loader = file_path #PyPDFLoader(file_path)
    #docs = loader.load()
    plain_text = markdown.markdown(file_path)
    docs = [Document(page_content=plain_text)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = rag_chain.invoke({"input": user_input})

    return results['answer']