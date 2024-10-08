__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


openai = st.secrets.db_credentials.openai
nvidia = st.secrets.db_credentials.nvidia

langchain_api = st.secrets.db_credentials.langchain_api

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]= langchain_api
os.environ['LANGCHAIN_PROJECT']= "Prep W Lervis"


#---------------------------------------------------------------------------------------------

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self.model.encode(d,batch_size=64).tolist() for d in documents]

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query])[0].tolist()

embedding_model = CustomEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#------------------------------------------------------------------------------------------------------------
llm = ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key = nvidia)

def pdf_rag(file_path, user_input):

    with open("temp_file.pdf", "wb") as f:
            f.write(file_path.getbuffer())

        
    loader = PyMuPDFLoader("temp_file.pdf")
    docs = loader.load()
    #plain_text = markdown.markdown(file_path)
    #docs = [Document(page_content=plain_text)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory="./chroma_langchain_db")

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for providing study materials for the provided job description. "
        "Use the following pieces of retrieved context which is job description to answer "
        "provide with study materials. Always answer questions or doubts related to job  description , and  provide study materials."
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
    if os.path.exists("temp_file.pdf"):
            os.remove("temp_file.pdf")

    return results['answer']
