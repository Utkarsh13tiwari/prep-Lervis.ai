#--------------------------------------------------------------------------------------------------
#imports
#----------------------------------------------
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
#from st_on_hover_tabs import on_hover_tabs
from langchain.callbacks import get_openai_callback
import time
from rag_app import webrag
import requests
from bs4 import BeautifulSoup
from query_agent import memory,agent_executor
from pdf_rag import pdf_rag
from tempfile import NamedTemporaryFile
import tempfile
from pathlib import Path

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
#--------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# APIwd
openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
api = st.secrets.db_credentials.api
search_engine = st.secrets.db_credentials.search_engine
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ICON_BLUE = "Lervis_logo.png"

st.logo(ICON_BLUE, icon_image=ICON_BLUE)

# Set the page configuration
st.set_page_config(
    page_title="Prep W Lervis",
    layout="wide",
    page_icon = ICON_BLUE
)  


st.markdown("""
<style>
	[data-testid="stDecoration"] {
		display: none;
	}

</style>""",
unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
                /* Hide the Streamlit header and menu */
                header {visibility: hidden;}
                /* Optionally, hide the footer */
                .streamlit-footer {display: none;}
                /* Hide your specific div class, replace class name with the one you identified */
                .st-emotion-cache-uf99v8 {display: none;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#---------------------------------------------------------------------------------------------------------------------------------------------

# Add custom CSS and HTML for the navbar
st.markdown(
    """
    <style>

    [alt=Logo] {
      height: 2.5rem;
    }

    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #0c0912;
        padding: 15px;
        position: fixed;
        top: 0;
        width: 100%;
        z-index: 1000;
        box-shadow: 0px 4px 2px -2px gray;
    }
    .navbar a {
        color: white;
        text-decoration: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .navbar a:hover {
        background-color: #333;
        border-radius: 5px;
    }
    .navbar .start-new-conversation {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .navbar .title {
        
        font-family: system-ui;
        color: white;
        font-size: 24px;
        font-weight: none;
        padding-left: 105px;
    }
    .content {
        margin-top: 80px; /* Adjust margin to account for fixed navbar */
    }
    .message-container {
        margin-bottom: 20px;
        width: 100%;
    }
    .message-container .user-message {
        text-align: right;
        padding: 10px;
        border-radius: 5px;
        background-color: #0c0912;
        margin-bottom: 20px;
    }
    .message-container .assistant-message {
        text-align: left;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .conversation-container {
        width: 80%;
        margin: 0 auto;
    }
    .sidebar-button {
        display: block;
        margin: 10px 0;
    }
    .streamlit-expanderHeader {
        font-size: 14px !important; /* Reduced font size */
    }
    .related-links-expander {
        max-height: 200px; /* Limit the height of the expander */
        overflow-y: auto;  /* Add scroll bar for long content */
        word-wrap: break-word; /* Break long words or URLs */
        overflow-wrap: break-word; /* Ensure long URLs are wrapped */
    }
    .related-links-expander a {
        display: block; /* Display links in block for better alignment */
        margin: 5px 0; /* Add some spacing between the links */
        color: #007bff; /* Blue color for links */
        text-decoration: none; /* Remove underline */
        font-size: 14px; /* Font size adjustment for links */
        max-width: 100%; /* Ensure the link does not exceed the container width */
        white-space: nowrap; /* Prevent the text from wrapping to the next line */
        overflow: hidden; /* Hide any overflowing content */
        text-overflow: ellipsis; /* Add an ellipsis for any overflowing text */
    }

    .st-emotion-cache-13k62yr {
        overflow-wrap: break-word;
    }
    .st-emotion-cache-13na8ym {
        width: 100% !important; /* Adjust width to desired size */
        display: inline-block !important; /* Ensure the expander is not full-width */
    }
    .MainMenu {
        visibility: hidden;
    }
    .st-emotion-cache-7tauuy {
        padding-left: 0rem;
    }
    .st-emotion-cache-qdbtli {
        width: 80%;
    }
    .content-row {
        display: flex;
        flex-direction: row;
        width: 100%;
    }
    .content-column {
        flex: 1;
        margin: 0 10px; /* Space between columns */
    }
    .checkbox-container label {
        color: white;
        margin-left: 5px;
    }
    .content-container {
        display: flex;
        flex-direction: column;
        width: 100%;
    }
    .checkbox-container label {
        color: white;
        margin-left: 5px;
    }
    .content-container {
        display: flex;
        flex-direction: column;
        width: 100%;
    }
    
    


    .checkbox-expander {
        position: fixed;
    }



    .st-emotion-cache-wud0ez {
    width: 202.172px;
    display: flex;
    -webkit-box-align: center;
    align-items: center;
    min-height: 2.5rem;
    padding-left: 20%;
    }
    .st-emotion-cache-1dkak6p {
    width: 161.5px;
    display: flex;
    -webkit-box-align: center;
    align-items: center;
    min-height: 2.5rem;
    padding-left: 10%;
    }

    .st-emotion-cache-wud0ez {
    width: 202.172px;
    display: flex;
    -webkit-box-align: center;
    align-items: center;
    min-height: 2.5rem;
    padding-left: 20%;
    }

    .st-emotion-cache-pb2n25 {
        position: fixed;
    }

    .st-emotion-cache-15f12so{
        position: fixed;
    }

    .st-emotion-cache-1xxg1wz {
        position: fixed;
    }

    ..st-emotion-cache-ocqkz7 {
    	position: absolute;
    }
    </style>
    <div class="navbar">
        <div class="title">Interview Preparation Assistant - Prep W Lervis</div>
        <div>
            <a href="https://www.linkedin.com/in/utkarsh-tiwari1313/">Connect</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 4, 1])
# Popover content for About
with col3:
    with st.popover("Usage"):
        st.markdown("""
	        <div style="padding: 10px; font-family: Arial, sans-serif;">
	            <h3 style="text-align: center;">About Prep W Lervis</h3>
	            <p>This agent helps individuals prepare for interviews by delivering pertinent information tailored to their inputs.</p>
	            <h4 Bot Usage Instructions:</h4>
	            <p>This agent's purpose is to:</p>
	            <ul style="margin-left: 20px;">
                	<li>Assist you with educational content related to interviews.</li>
                	<li>Answer your questions specifically about interview preparation topics.</li>
                	<li>Provide structured responses with relevant links and resources.</li>
                	<li>Offer soft skill training materials for interview readiness.</li>
                	<li>Implement RAG (Retrieve and Generate) functionality based on selected dropdown options if the RAG checkbox is checked.</li>
            	    </ul>
	            <h4>RAG Implementation:</h4>
	            <p>If you check the RAG checkbox, you will be able to implement RAG based on selected dropdown options.</p>
	        </div>
        """, unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Google Search ------------------------------------------------------------------------------------------------------------------------------------------------------------
def google_search(query, site=None):
    url = 'https://www.googleapis.com/customsearch/v1'
    if site:
        query = f"site:{site} {query}"
    
    params = {
        'q': query,
        'key': api,
        'cx': search_engine
    }
    response = requests.get(url, params=params)
    results = response.json()
    return [item.get('link') for item in results.get('items', [])]
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Frontend code. Initialising states ---------------------------------------------------------------------------------------------------------------------------------------
if 'conversations' not in st.session_state:
    st.session_state['conversations'] = {'Conversation 1': []}

if 'run' not in st.session_state:
    st.session_state.run = False

for conv_id in st.session_state['conversations']:
    if st.sidebar.button(conv_id):
        st.session_state['current_conversation'] = conv_id
                         
if st.sidebar.button("Start New Conversation"):
    new_conv_id = f"Conversation {len(st.session_state['conversations']) + 1}"
    st.session_state['conversations'][new_conv_id] = []
    st.session_state['current_conversation'] = new_conv_id  # Set the new conversation as the current one 

# Set the default current conversation
if 'current_conversation' not in st.session_state:
    st.session_state['current_conversation'] = 'Conversation 1'

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        memory.save_context({'input':message['user']},{'outputs':message['AI']})

#-------------------------------------------------------------------------------------------------------------------------------------
# General Functions section:
#------------------------------------------------------------------------------  
# Function to add new messages to the chat history
def add_message(role, content):
    st.session_state['chat_history'].append({"role": role, "content": content})
#------------------------------------------------------------------------------  

#---------------------------------------
def disable(value):
    st.session_state["disabled"] = value
#---------------------------------------

#------------------------------------------------------------------------------  
def fetch_title(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').get_text()
        return title
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return "No title available"
    except Exception as e:
        print(f"Error parsing title: {e}")
        return "No title available"
#------------------------------------------------------------------------------  
#-------------------------------------------------------------------------------------------------------------------------------------

# Use columns to center the conversation container and set width
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.write(f"### {st.session_state['current_conversation']} :")

    # Display the conversation history
    for message in st.session_state['conversations'][st.session_state['current_conversation']]:
        #message_class = "user-message" if message["isUser"] else "assistant-message"

        if message["isUser"]:
            message_class = "user-message"
            st.markdown(f'<div class="message-container"><p class="{message_class}">{message["text"]}</p></div>', unsafe_allow_html=True)

        elif not message["isUser"] and not message["related"]: 
            message_class = "assistant-message"
            st.write("### Agent Response:")
            st.markdown(f'<div class="message-container"><p class="{message_class}">{message["text"]}</p></div>', unsafe_allow_html=True)

        #with col3:
            #for message in st.session_state['conversations'][st.session_state['current_conversation']]:

#--------------------------------------------------------------------------------------------------------
# Input field for user query
user_input = st.chat_input("Enter your question or topic:",on_submit=disable, args=(True,))
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# Implement RAG or Query Agent:

with col1:
    st.markdown('<div class="checkbox-expander">', unsafe_allow_html=True)
    
    rag_checkbox = st.checkbox('RAG', key='rag_checkbox', disabled=st.session_state.get('disabled', False))
    
    if rag_checkbox:
        st.session_state['rag_expanded'] = True
    else:
        st.session_state['rag_expanded'] = False
    
    options = {
    'LeetCode': 'https://bishalsarang.github.io/Leetcode-Questions/out.html',
    'Job Description': None
    }

    if st.session_state['rag_expanded']:
        with st.expander("RAG Options", expanded=st.session_state.get('disabled', False)):
            selected_option = st.radio("Select an option:", list(options.keys()))
        if selected_option == 'Job Description':
            job_description = st.file_uploader('Upload Job Description (JD)', type="pdf")
            if job_description:
                import pymupdf4llm
                bytes_data = job_description.read()
                
                # create a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(bytes_data)
                    temp_file_path = tmp_file.name

                    job_des = pymupdf4llm.to_markdown(temp_file_path)
        else:
            link = options[selected_option] if selected_option else ""

    st.markdown('</div>', unsafe_allow_html=True)

#----------------------------------------------------------------------------------------------------------

if user_input:

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.write(f'<div class="message-container"><p class="user-message">{user_input}</p></div>', unsafe_allow_html=True)
    
    if rag_checkbox:
        with col2:
            st.write("#### Note that the RAG agent is still in development phase and may not work as expected")
        if selected_option == 'Job Description':
            with col2:
                with st.spinner("Reading PDF...."):
                    agent_response = pdf_rag(job_des, user_input=user_input)
        else:
            with col2:
                with st.spinner("Extracting...."):
                    agent_response = webrag(link, user_input)

    else:
        with get_openai_callback():
            with col2:
                with st.spinner("Agent"):
                    agent_response = agent_executor(user_input, st.session_state.chat_history)
        agent_response = agent_response["output"]

        message = {'user': user_input, 'AI': agent_response}
        st.session_state.chat_history.append(message)
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.write("### Agent Response:")
        def stream_data():
            for word in agent_response.split(" "):
                yield word + " "
                time.sleep(0.02)
        st.write_stream(stream_data)  # Typewriter effect
    

    # Save conversation history including related links
    with col2:
        st.session_state['conversations'][st.session_state['current_conversation']].append({"isUser": True, "text": user_input, "related": False})
        st.session_state['conversations'][st.session_state['current_conversation']].append({"isUser": False, "text": agent_response, "related": False})


    st.session_state["disabled"] = False 
    st.rerun()












