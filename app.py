import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import io
import os
import pandas as pd
import time
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bhasha Seva",
    page_icon="🗣️",
    layout="wide",
)

# --- PROJECT ASSETS & DATA ---

# Create a directory to save the collected data
if not os.path.exists("corpus_data"):
    os.makedirs("corpus_data")

# Path to the CSV file for storing metadata
CSV_PATH = "corpus_data/transcriptions.csv"

# Sample prompts in different Indian languages
PROMPTS = {
    "Hindi": [
        "नमस्ते, आपका स्वागत है।",
        "एकता में ही बल है।",
        "चलिए भारतीय भाषाओं के लिए एक बेहतर भविष्य का निर्माण करें।",
        "यह एक सुंदर दिन है और मैं बहुत खुश हूँ।",
        "कृपया इस वाक्य को स्पष्ट रूप से पढ़ें।",
    ],
    "Telugu": [
        "నమస్కారం, మీకు స్వాగతం.",
        "ఐకమత్యమే మహాబలం.",
        "భారతీయ భాషల కోసం ఒక మంచి భవిష్యత్తును నిర్మిద్దాం.",
        "ఇది ఒక అందమైన రోజు మరియు నేను చాలా సంతోషంగా ఉన్నాను.",
        "దయచేసి ఈ వాక్యాన్ని స్పష్టంగా చదవండి.",
    ],
    "Tamil": [
        "வணக்கம், வரவேற்கிறோம்.",
        "ஒற்றுமையே வலிமை.",
        "இந்திய மொழிகளுக்கு ஒரு சிறந்த எதிர்காலத்தை உருவாக்குவோம்.",
        "இது ஒரு அழகான நாள், நான் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்.",
        "தயவுசெய்து இந்த வாக்கியத்தை தெளிவாக படிக்கவும்.",
    ],
    "Bengali": [
        "নমস্কার, আপনাকে স্বাগত।",
        "একতাই বল।",
        "আসুন ভারতীয় ভাষার জন্য একটি ভালো ভবিষ্যৎ তৈরি করি।",
        "এটি একটি সুন্দর দিন এবং আমি খুব খুশি।",
        "অনুগ্রহ করে এই বাক্যটি স্পষ্টভাবে পড়ুন।",
    ]
}

# --- AUDIO PROCESSING CLASS ---

# This class will handle receiving audio frames from the browser
class AudioRecorder(AudioProcessorBase):
    def __init__(self) -> None:
        # A list to store audio frames
        self.audio_frames = []
        # A lock to prevent race conditions when accessing the list
        self._lock = True

    # This method is called for each audio frame
    def recv(self, frame):
        if self._lock:
            # Convert audio frame to a NumPy array of integers
            self.audio_frames.append(frame.to_ndarray())
        return frame

    # Method to get all collected audio frames
    def get_audio_frames(self):
        # Temporarily unlock to prevent new frames from being added
        self._lock = False
        # Concatenate all frames into a single NumPy array
        audio_array = np.concatenate(self.audio_frames)
        # Clear the list for the next recording
        self.audio_frames = []
        # Re-lock for the next recording session
        self._lock = True
        return audio_array

# --- HELPER FUNCTIONS ---

def initialize_csv():
    """Creates the CSV file with headers if it doesn't exist."""
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=["timestamp", "language", "prompt_text", "audio_filename", "transcribed_text"])
        df.to_csv(CSV_PATH, index=False)

def save_data(language, prompt, audio_bytes, transcription):
    """Saves the audio file and appends the metadata to the CSV."""
    timestamp = int(time.time())
    audio_filename = f"{timestamp}_{language}.wav"
    audio_filepath = os.path.join("corpus_data", audio_filename)

    # Save the audio file
    with open(audio_filepath, "wb") as f:
        f.write(audio_bytes)

    # Append data to CSV
    df = pd.read_csv(CSV_PATH)
    new_row = pd.DataFrame([{
        "timestamp": timestamp,
        "language": language,
        "prompt_text": prompt,
        "audio_filename": audio_filename,
        "transcribed_text": transcription
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

def transcribe_audio(audio_array, sample_rate):
    """Converts audio array to text using SpeechRecognition."""
    # Convert numpy array to AudioSegment
    audio_segment = AudioSegment(
        data=audio_array.tobytes(),
        sample_width=audio_array.dtype.itemsize,
        frame_rate=sample_rate,
        channels=1
    )

    # Export to a byte buffer in WAV format
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)

    # Use SpeechRecognition to transcribe
    r = sr.Recognizer()
    try:
        with sr.AudioFile(buffer) as source:
            audio_data = r.record(source)
        # Recognize using Google Web Speech API
        # Note: This requires an internet connection.
        # For better performance and privacy, consider models like Whisper.
        text = r.recognize_google(audio_data, language=st.session_state.lang_code)
        return text, buffer.getvalue()
    except sr.UnknownValueError:
        return "AI could not understand the audio. Please try again.", None
    except sr.RequestError as e:
        return f"Could not request results from the AI service; {e}", None
    except Exception as e:
        return f"An unexpected error occurred: {e}", None

# --- STREAMLIT APP UI ---

st.title("🗣️ Bhasha Seva – Indian Language Corpus Collector")
st.markdown("""
Welcome! This app helps build a rich dataset for Indian languages.
Your contribution will improve AI tools like speech recognition and translation for everyone.
**How it works:**
1.  Select a language.
2.  Click **Start Recording** and read the provided sentence aloud.
3.  Click **Stop Recording** when you're done.
4.  The app will transcribe your voice and save the data.
""")

st.divider()

# Initialize CSV file
initialize_csv()

# --- MAIN APP LOGIC ---

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Select Your Language")
    language_map = {"Hindi": "hi-IN", "Telugu": "te-IN", "Tamil": "ta-IN", "Bengali": "bn-IN"}
    selected_language = st.selectbox("Choose a language to contribute to:", list(language_map.keys()))
    st.session_state.lang_code = language_map[selected_language]

    # Display a random prompt for the selected language
    if 'prompt' not in st.session_state or st.session_state.get('lang') != selected_language:
        st.session_state.prompt = random.choice(PROMPTS[selected_language])
        st.session_state.lang = selected_language

    st.subheader("2. Read This Sentence Aloud")
    st.info(f"**Prompt:** \"{st.session_state.prompt}\"")

    # Button to get a new prompt
    if st.button("Get a New Sentence"):
        st.session_state.prompt = random.choice(PROMPTS[selected_language])
        st.rerun()


with col2:
    st.subheader("3. Record Your Voice")

    # webrtc_streamer for audio recording
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"audio": True, "video": False},
        send_audio_frame_rate=16000 # Set sample rate
    )

    st.subheader("4. Review and Submit")

    if not webrtc_ctx.state.playing:
        st.warning("Click 'Start Recording' above to begin.")
    
    if webrtc_ctx.audio_processor:
        # Check if there are audio frames collected
        if st.button("Process My Voice Recording"):
            with st.spinner("AI is transcribing your voice... Please wait."):
                audio_processor = webrtc_ctx.audio_processor
                audio_array = audio_processor.get_audio_frames()

                if audio_array is not None and len(audio_array) > 0:
                    sample_rate = webrtc_ctx.send_audio_frame_rate
                    
                    # Transcribe the audio
                    transcription, audio_bytes = transcribe_audio(audio_array, sample_rate)

                    st.session_state.transcription = transcription
                    st.session_state.audio_bytes = audio_bytes
                else:
                    st.error("No audio was recorded. Please try again.")

    if 'transcription' in st.session_state:
        st.markdown("**Your Original Prompt:**")
        st.write(f"> _{st.session_state.prompt}_")
        st.markdown("**AI-Generated Transcription:**")
        st.write(f"> _{st.session_state.transcription}_")

        if st.session_state.audio_bytes:
            st.audio(st.session_state.audio_bytes, format='audio/wav')
            if st.button("✅ Looks Good! Save Contribution"):
                save_data(selected_language, st.session_state.prompt, st.session_state.audio_bytes, st.session_state.transcription)
                st.success("Thank you! Your contribution has been saved.")
                # Clear state for next recording
                del st.session_state.transcription
                del st.session_state.audio_bytes
                # Get a new prompt automatically
                st.session_state.prompt = random.choice(PROMPTS[selected_language])
                st.rerun()

st.divider()
st.markdown("Made with ❤️ for the future of Indian languages.")
