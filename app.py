import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Bhasha Seva – Indian Language Corpus Collector",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Project Title and Description ---
st.title("Bhasha Seva – Indian Language Corpus Collector 🇮🇳")
st.markdown("""
Welcome to **Bhasha Seva**! This platform is dedicated to building a rich and diverse corpus of Indian languages. 
Your contributions will help in the development of language technologies like machine translation, speech recognition, and more for our languages.
""")
st.markdown("---")


# --- Data Storage Setup ---
DATA_FILE = "language_corpus.csv"

def get_data():
    """Reads the collected data from the CSV file."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Create the file with headers if it doesn't exist
        df = pd.DataFrame(columns=["Language", "Text", "Age", "Gender", "Region/Dialect"])
        df.to_csv(DATA_FILE, index=False)
        return df

def save_data(new_data):
    """Saves new data to the CSV file."""
    df = get_data()
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)


# --- Sidebar for Metadata ---
st.sidebar.header("About You")
st.sidebar.markdown("This information helps us build a more balanced dataset.")
age = st.sidebar.slider("Select your age:", 10, 100, 25)
gender = st.sidebar.selectbox("Select your gender:", ["Male", "Female", "Other", "Prefer not to say"])
region = st.sidebar.text_input("Your Region/Dialect (e.g., Awadhi, Malwa, Konkan)")


# --- Main Application ---
st.header("Contribute Your Language")

# List of Indian languages
indian_languages = [
    "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada",
    "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi",
    "Nepali", "Odia", "Punjabi", "Sanskrit", "Santali", "Sindhi",
    "Tamil", "Telugu", "Urdu"
]

selected_language = st.selectbox("1. Select a Language:", indian_languages)

st.write(f"You have selected: **{selected_language}**")

# Text area for input
text_input = st.text_area(f"2. Please enter a sentence or a paragraph in {selected_language}:", height=200)

# Submit button
if st.button("Submit Contribution", type="primary"):
    if text_input and region:
        new_entry = pd.DataFrame({
            "Language": [selected_language],
            "Text": [text_input],
            "Age": [age],
            "Gender": [gender],
            "Region/Dialect": [region]
        })
        save_data(new_entry)
        st.success("Thank you for your contribution! Your entry has been saved.")
        st.balloons()
    else:
        st.warning("Please make sure to fill in the text and your region/dialect before submitting.")

# --- Display Collected Data ---
st.markdown("---")
st.header("Collected Corpus Data")

corpus_df = get_data()

if not corpus_df.empty:
    st.dataframe(corpus_df.style.set_properties(**{'text-align': 'left'}))

    # --- Download Button ---
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(corpus_df)

    st.download_button(
       label="Download data as CSV",
       data=csv,
       file_name='language_corpus.csv',
       mime='text/csv',
    )
else:
    st.info("No data has been collected yet. Be the first to contribute!")

# --- Footer ---
st.markdown("---")
st.markdown("""
*Made with ❤️ for Indian Languages.*
""")
