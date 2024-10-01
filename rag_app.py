__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb.api
chromadb.api.client.SharedSystemClient.clear_system_cache()

import os
import openai
import streamlit as st
import requests
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from bs4 import BeautifulSoup
import bs4
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.callbacks import get_openai_callback
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import langsmith


openai = st.secrets.db_credentials.openai
nvidia = st.secrets.db_credentials.nvidia

langchain_api = st.secrets.db_credentials.langchain_api
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=langchain_api
os.environ['LANGCHAIN_PROJECT']= "Prep W Lervis"


try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key = nvidia)
except ImportError:
    raise ImportError("ChatNVIDIA is not available. Please check your installation.")

def webrag(link, user_input):

    loader = WebBaseLoader(
        web_paths=(link,),
    )
    docs = loader.load()


    # Ensure that documents are loaded
    if not docs:
        st.error("No documents were loaded from the provided link.")
        return None

    # Process documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create vector store from the documents
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key = openai), persist_directory="./chroma_langchain_db")

    # Retrieve and generate using the relevant snippets
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    response = rag_chain.invoke(user_input)

    return response
