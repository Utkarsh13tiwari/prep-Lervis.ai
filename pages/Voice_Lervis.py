import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.memory import ConversationBufferWindowMemory

import edge_tts
import asyncio
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

import whisper
import torch 
#import sounddevice as sd  # Make sure sounddevice is correctly imported
import numpy as np
from scipy.io.wavfile import write
import tempfile
import threading
import os
from streamlit_mic_recorder import mic_recorder, speech_to_text
import streamlit_mic_recorder
from langchain_groq import ChatGroq
from langchain.chains import LLMChain

# Set environment variables
openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
api = st.secrets.db_credentials.api
search_engine = st.secrets.db_credentials.search_engine
GROQ_API_KEY = st.secrets.db_credentials.GROQ_API_KEY

langchain_api = st.secrets.db_credentials.langchain_api
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=langchain_api
os.environ['LANGCHAIN_PROJECT']= "Prep W Lervis"


ICON_BLUE = "Lervis_logo.png"

st.logo(ICON_BLUE, icon_image=ICON_BLUE)


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

col1, col2, col3 = st.columns([1, 4, 1])
# Popover content for About
with col3:
    with st.popover("Usage"):
        st.markdown("""
	        <div style="padding: 10px; font-family: Arial, sans-serif;">
	            <p>This voice assistant acts as your personal virtual interviewer. It tailors interview questions based on the job description, and the follow-up questions are determined by your responses.</p>
	        </div>
        """, unsafe_allow_html=True)
	    
# Initialize the agent and memory
search = TavilySearchAPIWrapper(tavily_api_key=tavily)
tavily_tool = TavilySearchResults(api_wrapper=search)
tools = [tavily_tool]

memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=10)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI interviewer. Based on the job description provided,"
        "create relevant interview questions. Ask user, questions one by one, one each time and then let user reply after every asked question first."
        "Dont ask multiple questions at a time. Make sure you build up upon the user previouse answers."
        "Apart from the interview related or question you will not answer anything."),
        ("user", "{JD}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ]
)

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY,
    # other params...
)

#agent = create_tool_calling_agent(llm, tools, prompt_template)
agent_executor = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True,
    memory=memory
)
#agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, stream_runnable=False)
#--------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------
if 'is_recording' not in st.session_state:
    st.session_state['is_recording'] = False

if 'text_received' not in st.session_state:
    st.session_state.text_received = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        memory.chat_memory.add_user_message(message['user'])
        memory.chat_memory.add_ai_message(message['AI'])
        #memory.save_context({'input':message['user']},{'outputs':message['AI']})

#--------------------------------------------------------------------------------------------------------------------------

async def text_to_speech_edge(text):

    tts = edge_tts.Communicate(text, "en-US-JennyNeural")  
    audio_fp = BytesIO()

    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            audio_fp.write(chunk["data"])

    audio_fp.seek(0)

    audio = AudioSegment.from_file(audio_fp, format="mp3")
    wav_fp = BytesIO()
    audio.export(wav_fp, format="wav") 
    wav_fp.seek(0) 
    return wav_fp

def text_to_speech(text):
    wav_audio = asyncio.run(text_to_speech_edge(text))
    st.audio(wav_audio, autoplay=True ,format='audio/wav') 
    return

#st.set_page_config(page_title="Virtual AI Interviewer", layout="wide")
row1 = st.columns([1, 15, 1])
row2 = st.columns([1, 15, 1])
#col1, col2, col3 = st.columns([1, 3, 1])
container = st.container(border=True)
with row1[1]:
    st.title("Virtual AI Interviewer")

    # User inputs the job description
    JD = st.text_area("Enter Job Description (JD)")

    # Store the state for the first question
    if "first_question_asked" not in st.session_state:
        st.session_state.first_question_asked = False

    # Button to start the interview
    if st.button("Start Interview"):
        if JD and not st.session_state.first_question_asked:
            # Generate the first question
            with st.container():
                with st.spinner("Working......"):
                    question = agent_executor.invoke({"JD": JD,"input": "", "chat_history": st.session_state.chat_history})
                question = question['text']
                st.session_state.first_question = question
                st.write(f"AI Interviewer: {question}")
                message = {'user': JD, 'AI': question}
                #print(memory)
            #memory.save_context({'JD':message['user']},{'outputs':message['AI']})
                st.session_state.chat_history.append(message)
            text_to_speech(question)
            st.session_state.first_question_asked = True
        elif not JD:
            st.write("Please provide a job description to start the interview.")

    if st.session_state.first_question_asked:
        st.write("Start replying to the Questions by clicking Start-recording button:")

        # Fixed button for speech-to-text
        with row2[1]:
            text = streamlit_mic_recorder.speech_to_text(language='en', use_container_width=False, just_once=True, key='STT')

        with row1[1]:
            if text:
                st.session_state.text_received.append(text)

            for text in st.session_state.text_received:
                st.text(text)

            if text is not None:
                user_response = text
                with st.spinner("Working......"):
                    next_question = agent_executor.invoke({"JD": JD, "input": user_response, "chat_history": st.session_state.chat_history})
                    next_question = next_question['text']
                message = {'user': user_response, 'AI': next_question}
                st.session_state.chat_history.append(message)
                st.write(f"AI Interviewer: {next_question}")
                text_to_speech(next_question)

