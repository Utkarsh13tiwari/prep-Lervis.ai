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
import time
#from langchain_google_vertexai import ChatVertexAI


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# API
openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
api = st.secrets.db_credentials.api
search_engine = st.secrets.db_credentials.search_engine

langchain_api = st.secrets.db_credentials.langchain_api

export LANGCHAIN_TRACING_V2 = true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY = langchain_api
export LANGCHAIN_PROJECT= "Prep W Lervis"
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Creating Agent ------------------------------------------------------------------------------------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant named Prep W Lervis. "
            "Your purpose is to assist people who are preparing for their interviews or want to learn specific topics as part of their interview preparation process. "
            "You will not answer any question apart from educational contents and questions related to interview preparation. "
            "You will help the user prepare for the topics they mention, provide them with relevant links, "
            "and present the content in a structured manner. "
            "You can also provide users with educational content and soft skill training materials.\n\n"
            "Give all the information needed in an exhaustive manner. "
            "Please format your responses with headings, bullet points, and clear sections. Make use of bullet points, headings, and better formatting.\n\n"
            "Add proper gaps between the points, start different points from different lines, make multiple paragraphs and always respond in an exhaustive manner.",
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


openai_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai)
search = TavilySearchAPIWrapper(tavily_api_key=tavily)
tavily_tool = TavilySearchResults(api_wrapper=search)
tools = [tavily_tool]

functions = [format_tool_to_openai_function(t) for t in tools]
llm_with_tools = openai_model.bind_functions(functions=functions)
memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=1)


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
def agent_executor(user_input,chat_history):
    agent_exe=AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True ,stream_runnable=False)
    agent_response=agent_exe.invoke({"input": user_input, "chat_history":chat_history})
    return agent_response
