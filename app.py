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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# API
openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
api = st.secrets.db_credentials.api
search_engine = st.secrets.db_credentials.search_engine
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Set the page configuration
st.set_page_config(
    page_title="Prep W Lervis",
    layout="wide"
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
        color: white;
        font-size: 24px;
        font-weight: bold;
        padding-left: 55px;
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
    </style>
    <div class="navbar">
        <div class="title">Interview Preparation Assistant - Prep W Lervis</div>
        <div>
            <a href="#contact">Contact</a>
            <a href="#Profile">Profile</a>
        </div>
    </div>
    <div class="content">
    """,
    unsafe_allow_html=True,
)


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

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True ,stream_runnable=False)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Google Search ------------------------------------------------------------------------------------------------------------------------------------------------------------
def google_search(query):
    url = 'https://www.googleapis.com/customsearch/v1'
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

if 'processing' not in st.session_state:
    st.session_state['processing'] = False

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


# Use columns to center the conversation container and set width
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.write(f"### {st.session_state['current_conversation']} :")

    # Display the conversation history
    for message in st.session_state['conversations'][st.session_state['current_conversation']]:
        message_class = "user-message" if message["isUser"] else "assistant-message"
        st.markdown(f'<div class="message-container"><p class="{message_class}">{message["text"]}</p></div>', unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        memory.save_context({'input':message['user']},{'outputs':message['AI']})

# Function to add new messages to the chat history
def add_message(role, content):
    st.session_state['chat_history'].append({"role": role, "content": content})


    
# Input field for user query
user_input = st.chat_input("Enter your question or topic:")



#with col1:
#    st.markdown('<div class="checkbox-expander">', unsafe_allow_html=True)
    
    # Disable the checkbox if processing is True
#    rag_checkbox = st.checkbox('RAG', key='rag_checkbox', value=st.session_state.get('rag_expanded', False), disabled=st.session_state.processing)
    
    # Update the expander state based on checkbox value
#    if rag_checkbox:
#        st.session_state['rag_expanded'] = True
#    else:
#        st.session_state['rag_expanded'] = False
    
    # Display expander if checkbox is checked
#    if st.session_state['rag_expanded']:
#        with st.expander("RAG Options", expanded=True):
            # Display checkboxes inside the expander
#            st.checkbox('Option 1')
#            st.checkbox('Option 2')
#            st.checkbox('Option 3')
#            st.checkbox('Option 4')
#            st.checkbox('Option 5')
#    st.markdown('</div>', unsafe_allow_html=True)



if user_input and not st.session_state['processing']:
    st.session_state.processing = True  # Set processing flag to True
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.write(f'<div class="message-container"><p class="user-message">{user_input}</p></div>', unsafe_allow_html=True)
    
    # Process user input with the agent
    with get_openai_callback():
        agent_response = agent_executor.invoke({"input": user_input, "chat_history":st.session_state.chat_history})
    agent_response = agent_response["output"]

    message = {'user': user_input, 'AI': agent_response}
    st.session_state.chat_history.append(message)

    # Perform Google search for related links
    related_links = google_search(user_input)

    # Display the response
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.write("### Agent Response:")
        def stream_data():
            for word in agent_response.split(" "):
                yield word + " "
                time.sleep(0.02)
        st.write_stream(stream_data) #-----------------------------------Type writer effect
    
    # Display related links in a dropdown
    with col3:
        st.write("### Related Links:")
        with st.expander("Google Search", expanded=False):
            st.markdown('<div class="related-links-expander">', unsafe_allow_html=True)
            for link in related_links:
                st.write(f"- [{link}]({link})")
            st.markdown('</div>', unsafe_allow_html=True)

    # Save conversation history
    with col2:
        st.session_state['conversations'][st.session_state['current_conversation']].append({"isUser": True, "text": user_input})
        st.session_state['conversations'][st.session_state['current_conversation']].append({"isUser": False, "text": agent_response})
     
    st.session_state.processing = False  # Set processing flag to False after processing (Must be used for rag)

