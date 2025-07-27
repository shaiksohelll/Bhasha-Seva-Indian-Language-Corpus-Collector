import streamlit as st
from googletrans import Translator, LANGUAGES
import speech_recognition as sr
from pydub import AudioSegment
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Bhasha Seva – Translator",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- App Title and Description ---
st.title("Bhasha Seva – Translator 🗣️")
st.markdown("""
This tool translates Telugu text and audio into English. 
You can either type the text directly or provide an audio input in Telugu.
""")
st.markdown("---")

# --- Initialize Translator ---
# Create a translator object. The 'proxies' parameter can be used if you are behind a proxy.
# For most users, this is not necessary.
translator = Translator()

# --- Functions for Translation and Speech Recognition ---

def translate_text(text, dest_lang):
    """
    Translates the given text to the destination language.
    """
    try:
        # Detect the source language of the input text
        detected_lang = translator.detect(text).lang
        # Translate the text to the destination language
        translated = translator.translate(text, src=detected_lang, dest=dest_lang)
        # Return the translated text and the detected source language
        return translated.text, detected_lang
    except Exception as e:
        # Handle exceptions during translation
        st.error(f"An error occurred during translation: {e}")
        return None, None

def transcribe_audio(audio_bytes):
    """
    Transcribes the given audio bytes to text using Google's Speech Recognition.
    """
    # Initialize the recognizer
    r = sr.Recognizer()
    try:
        # Convert audio bytes to a format that speech_recognition can handle
        # The audio is first loaded into an AudioSegment object
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # Export the audio segment to a WAV format in memory
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        # Use the WAV data from memory as the audio source
        with sr.AudioFile(wav_io) as source:
            # Record the audio data from the source
            audio_data = r.record(source)
            # Recognize the speech in the audio data using Google's engine
            # The language is specified as 'te-IN' for Telugu (India)
            text = r.recognize_google(audio_data, language='te-IN')
            return text
    except sr.UnknownValueError:
        # Handle cases where the speech is unintelligible
        st.warning("Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        # Handle errors with the speech recognition service
        st.error(f"Could not request results from the speech recognition service; {e}")
        return None
    except Exception as e:
        # Handle other exceptions
        st.error(f"An error occurred during audio processing: {e}")
        return None

# --- UI for Input Selection ---
input_method = st.radio(
    "Choose your input method:",
    ("Text Input", "Audio Input")
)

st.markdown("---")

# --- Processing based on Input Method ---

if input_method == "Text Input":
    st.subheader("Translate Telugu Text to English")
    # Text area for user input
    text_to_translate = st.text_area("Enter Telugu text here:", height=150)

    if st.button("Translate Text", type="primary"):
        if text_to_translate:
            with st.spinner("Translating..."):
                # Translate the input text
                translated_text, detected_lang = translate_text(text_to_translate, 'en')
                if translated_text:
                    st.success("Translation Complete!")
                    # Display the results
                    st.markdown("### Results")
                    st.write(f"**Original (Telugu):** {text_to_translate}")
                    st.write(f"**Translation (English):** {translated_text}")
        else:
            st.warning("Please enter some text to translate.")

elif input_method == "Audio Input":
    st.subheader("Translate Telugu Audio to English")
    # Audio input widget
    audio_file = st.file_uploader("Upload a Telugu audio file (WAV, MP3):", type=['wav', 'mp3', 'ogg'])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        
        if st.button("Transcribe and Translate Audio", type="primary"):
            with st.spinner("Processing Audio... Please Wait."):
                # Read the audio file bytes
                audio_bytes = audio_file.read()
                
                # Transcribe the audio to get Telugu text
                telugu_text = transcribe_audio(audio_bytes)

                if telugu_text:
                    st.info(f"**Transcribed Telugu Text:** {telugu_text}")
                    with st.spinner("Translating Text..."):
                        # Translate the transcribed text to English
                        english_text, _ = translate_text(telugu_text, 'en')
                        if english_text:
                            st.success("Translation Complete!")
                            # Display the final results
                            st.markdown("### Results")
                            st.write(f"**Subtitles (Telugu):** {telugu_text}")
                            st.write(f"**Translation (English):** {english_text}")

# --- Footer ---
st.markdown("---")
st.markdown("*Powered by Google Translate and Speech Recognition.*")
