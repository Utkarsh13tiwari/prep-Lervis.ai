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
from langchain.agents import AgentExecutor
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
import sounddevice as sd  # Make sure sounddevice is correctly imported
import numpy as np
from scipy.io.wavfile import write
import tempfile
import threading
import os

# Set environment variables
openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
api = st.secrets.db_credentials.api
search_engine = st.secrets.db_credentials.search_engine


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

# Initialize the agent and memory
openai_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai)
search = TavilySearchAPIWrapper(tavily_api_key=tavily)
tavily_tool = TavilySearchResults(api_wrapper=search)
tools = [tavily_tool]

# Format tools to OpenAI functions
functions = [format_tool_to_openai_function(t) for t in tools]
llm_with_tools = openai_model.bind_functions(functions=functions)

# Define the memory and prompt template
memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='JD', k=1)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI interviewer. Based on the job description provided, create relevant interview questions. Ask user, questions one by one, and then let user reply after every asked question first."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{JD}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent
agent = (
    {
        "JD": lambda x: x["JD"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt_template
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Initialize the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

#--------------------------------------------------------------------------------------------------------------------------
if 'is_recording' not in st.session_state:
    st.session_state['is_recording'] = False



#--------------------------------------------------------------------------------------------------------------------------
# Helper functions for TTS and STT
async def text_to_speech_edge(text):
    # Initialize the TTS object and output stream
    tts = edge_tts.Communicate(text, "en-US-JennyNeural")  # Specify the voice
    audio_fp = BytesIO()

    # Stream the generated audio to the BytesIO object
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            audio_fp.write(chunk["data"])

    # Make sure the stream is at the beginning for reading
    audio_fp.seek(0)

    # Read the audio from BytesIO using pydub
    audio = AudioSegment.from_file(audio_fp, format="mp3")
    return audio

def text_to_speech(text):
    # Run the async function and play the audio
    audio = asyncio.run(text_to_speech_edge(text))
    play(audio)

# Function to record audio

def record_audio(duration=5, fs=44100):
    """Record audio for a given duration and sampling rate."""
    st.write("Listening...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    st.write("Recording complete.")
    return recording, fs

# Function to convert speech to text using Whisper
def speech_to_text():
    try:
        # Record audio
        audio_data, fs = record_audio()

        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            write(temp_audio_file.name, fs, audio_data)
            temp_filename = temp_audio_file.name

        # Load Whisper model
        model = whisper.load_model("base")

        # Transcribe the recorded audio
        result = model.transcribe(temp_filename)

        # Clean up the temporary audio file
        os.remove(temp_filename)

        # Return the transcribed text
        return result["text"]

    except Exception as e:
        st.write(f"Microphone error or Whisper error: {str(e)}")
        return "Microphone is not accessible or recognized."


# Streamlit UI
#st.set_page_config(page_title="Virtual AI Interviewer", layout="wide")

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
        question = agent_executor.invoke({"JD": JD})['output']
        st.session_state.first_question = question
        st.write(f"AI Interviewer: {question}")
        text_to_speech(question)
        st.session_state.first_question_asked = True
    elif not JD:
        st.write("Please provide a job description to start the interview.")

# Button for the user to provide a response
if st.session_state.first_question_asked and st.button("Reply"):
    user_response = speech_to_text()
    st.write(f"You: {user_response}")

    # Generate the next question based on the user's input
    next_question = agent_executor.invoke({"JD": JD, "input": user_response})['output']
    st.write(f"AI Interviewer: {next_question}")
    text_to_speech(next_question)
