import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime
import numpy as np

# Try to import optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Install with: pip install plotly")

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    st.warning("langdetect not available. Install with: pip install langdetect")

# Indian language mapping
INDIAN_LANGUAGES = {
    'hi': 'Hindi (हिन्दी)',
    'bn': 'Bengali (বাংলা)', 
    'te': 'Telugu (తెలుగు)',
    'mr': 'Marathi (मराठी)',
    'ta': 'Tamil (தமிழ்)',
    'ur': 'Urdu (اردو)',
    'gu': 'Gujarati (ગુજરાતી)',
    'kn': 'Kannada (ಕನ್ನಡ)',
    'ml': 'Malayalam (മലയാളം)',
    'pa': 'Punjabi (ਪੰਜਾਬੀ)',
    'or': 'Odia (ଓଡ଼ିଆ)',
    'as': 'Assamese (অসমীয়া)',
    'ne': 'Nepali (नेपाली)',
    'sa': 'Sanskrit (संस्कृत)'
}

TEXT_CATEGORIES = [
    "News & Journalism",
    "Literature & Poetry", 
    "Social Media",
    "Government Documents",
    "Academic Papers",
    "Religious Texts",
    "Folk Tales & Stories",
    "Educational Content",
    "Other"
]

class SimpleLanguageDetector:
    """Simple language detector using basic patterns"""
    
    def __init__(self):
        # Basic script detection patterns
        self.script_patterns = {
            'hi': r'[\u0900-\u097F]',  # Devanagari
            'bn': r'[\u0980-\u09FF]',  # Bengali
            'te': r'[\u0C00-\u0C7F]',  # Telugu
            'ta': r'[\u0B80-\u0BFF]',  # Tamil
            'mr': r'[\u0900-\u097F]',  # Devanagari (same as Hindi)
            'gu': r'[\u0A80-\u0AFF]',  # Gujarati
            'kn': r'[\u0C80-\u0CFF]',  # Kannada
            'ml': r'[\u0D00-\u0D7F]',  # Malayalam
            'pa': r'[\u0A00-\u0A7F]',  # Gurmukhi
            'or': r'[\u0B00-\u0B7F]',  # Odia
            'ur': r'[\u0600-\u06FF]',  # Arabic/Urdu
        }
    
    def detect_language(self, text):
        """Simple script-based language detection"""
        import re
        
        if not text:
            return "unknown", "Unknown"
        
        # Count characters for each script
        script_scores = {}
        total_chars = len(re.findall(r'\S', text))
        
        if total_chars == 0:
            return "unknown", "Unknown"
        
        for lang_code, pattern in self.script_patterns.items():
            matches = len(re.findall(pattern, text))
            if matches > 0:
                script_scores[lang_code] = matches / total_chars
        
        if script_scores:
            best_lang = max(script_scores, key=script_scores.get)
            if script_scores[best_lang] > 0.1:  # At least 10% of characters
                return best_lang, INDIAN_LANGUAGES.get(best_lang, best_lang)
        
        return "unknown", "Unknown"

class CorpusManager:
    def __init__(self):
        self.init_database()
        if LANGDETECT_AVAILABLE:
            self.detector = None  # Use langdetect
        else:
            self.detector = SimpleLanguageDetector()
    
    def init_database(self):
        """Initialize SQLite database for corpus storage"""
        conn = sqlite3.connect('bhasha_seva_corpus.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corpus_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                language_code TEXT,
                language_name TEXT,
                source TEXT,
                category TEXT,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                char_count INTEGER,
                contributor TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS language_stats (
                language_code TEXT PRIMARY KEY,
                language_name TEXT,
                total_texts INTEGER,
                total_words INTEGER,
                total_chars INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_language(self, text):
        """Detect language of the given text"""
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(text)
                if detected_lang in INDIAN_LANGUAGES:
                    return detected_lang, INDIAN_LANGUAGES[detected_lang]
                else:
                    return detected_lang, detected_lang.title()
            except LangDetectException:
                return "unknown", "Unknown"
        else:
            return self.detector.detect_language(text)
    
    def add_text_to_corpus(self, text, source, category, contributor, manual_lang=None):
        """Add text to corpus database"""
        if manual_lang and manual_lang in INDIAN_LANGUAGES:
            lang_code = manual_lang
            lang_name = INDIAN_LANGUAGES[manual_lang]
        else:
            lang_code, lang_name = self.detect_language(text)
        
        word_count = len(text.split())
        char_count = len(text)
        
        conn = sqlite3.connect('bhasha_seva_corpus.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO corpus_data 
            (text, language_code, language_name, source, category, word_count, char_count, contributor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (text, lang_code, lang_name, source, category, word_count, char_count, contributor))
        
        # Update language statistics
        cursor.execute('''
            INSERT OR REPLACE INTO language_stats 
            (language_code, language_name, total_texts, total_words, total_chars)
            VALUES (?, ?, 
                COALESCE((SELECT total_texts FROM language_stats WHERE language_code = ?), 0) + 1,
                COALESCE((SELECT total_words FROM language_stats WHERE language_code = ?), 0) + ?,
                COALESCE((SELECT total_chars FROM language_stats WHERE language_code = ?), 0) + ?
            )
        ''', (lang_code, lang_name, lang_code, lang_code, word_count, lang_code, char_count))
        
        conn.commit()
        conn.close()
        
        return lang_code, lang_name
    
    def get_corpus_stats(self):
        """Get corpus statistics"""
        conn = sqlite3.connect('bhasha_seva_corpus.db')
        
        # Overall stats
        total_texts_result = pd.read_sql_query("SELECT COUNT(*) as count FROM corpus_data", conn)
        total_texts = total_texts_result.iloc[0]['count'] if not total_texts_result.empty else 0
        
        total_words_result = pd.read_sql_query("SELECT SUM(word_count) as total FROM corpus_data", conn)
        total_words = total_words_result.iloc[0]['total'] if not total_words_result.empty and total_words_result.iloc[0]['total'] else 0
        
        total_chars_result = pd.read_sql_query("SELECT SUM(char_count) as total FROM corpus_data", conn)
        total_chars = total_chars_result.iloc[0]['total'] if not total_chars_result.empty and total_chars_result.iloc[0]['total'] else 0
        
        # Language-wise stats
        lang_stats = pd.read_sql_query("SELECT * FROM language_stats ORDER BY total_words DESC", conn)
        
        # Category-wise stats
        category_stats = pd.read_sql_query("""
            SELECT category, COUNT(*) as count, SUM(word_count) as words 
            FROM corpus_data 
            GROUP BY category 
            ORDER BY count DESC
        """, conn)
        
        conn.close()
        
        return {
            'total_texts': total_texts,
            'total_words': int(total_words),
            'total_chars': int(total_chars),
            'language_stats': lang_stats,
            'category_stats': category_stats
        }
    
    def search_corpus(self, query, language=None, category=None):
        """Search corpus data"""
        conn = sqlite3.connect('bhasha_seva_corpus.db')
        
        sql = "SELECT * FROM corpus_data WHERE text LIKE ?"
        params = [f"%{query}%"]
        
        if language and language != "All":
            sql += " AND language_code = ?"
            params.append(language)
        
        if category and category != "All":
            sql += " AND category = ?"
            params.append(category)
        
        sql += " ORDER BY date_added DESC LIMIT 100"
        
        results = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        
        return results
    
    def export_corpus(self, language=None, format='csv'):
        """Export corpus data"""
        conn = sqlite3.connect('bhasha_seva_corpus.db')
        
        if language and language != "All":
            df = pd.read_sql_query(
                "SELECT * FROM corpus_data WHERE language_code = ? ORDER BY date_added DESC", 
                conn, params=[language]
            )
        else:
            df = pd.read_sql_query("SELECT * FROM corpus_data ORDER BY date_added DESC", conn)
        
        conn.close()
        
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'json':
            return df.to_json(orient='records', indent=2, ensure_ascii=False)
        else:
            return df

def create_simple_chart(data, chart_type="bar"):
    """Create simple charts without plotly"""
    if chart_type == "bar" and not data.empty:
        st.bar_chart(data.set_index(data.columns[0]))
    elif chart_type == "line" and not data.empty:
        st.line_chart(data.set_index(data.columns[0]))

def main():
    st.set_page_config(
        page_title="Bhasha Seva - Indian Language Corpus Collector",
        page_icon="🇮🇳",
        layout="wide"
    )
    
    # Initialize corpus manager
    corpus_manager = CorpusManager()
    
    # Header
    st.title("🇮🇳 Bhasha Seva - Indian Language Corpus Collector")
    st.markdown("*Preserving and collecting Indian language texts for research and development*")
    
    # Show installation status
    with st.expander("📋 Installation Status"):
        st.write("**Required Dependencies:**")
        st.write(f"✅ Streamlit: Available")
        st.write(f"✅ Pandas: Available") 
        st.write(f"✅ SQLite: Available")
        st.write(f"✅ NumPy: Available")
        
        st.write("**Optional Dependencies:**")
        st.write(f"{'✅' if PLOTLY_AVAILABLE else '❌'} Plotly: {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")
        st.write(f"{'✅' if LANGDETECT_AVAILABLE else '❌'} LangDetect: {'Available' if LANGDETECT_AVAILABLE else 'Not Available'}")
        
        if not PLOTLY_AVAILABLE or not LANGDETECT_AVAILABLE:
            st.warning("Some features may be limited. Install missing packages for full functionality.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📝 Add Text", 
        "📊 Corpus Statistics", 
        "🔍 Search Corpus", 
        "📥 Export Data"
    ])
    
    if page == "📝 Add Text":
        st.header("Add Text to Corpus")
        
        # Input methods
        input_method = st.radio("Choose input method:", 
                               ["Manual Text Entry", "File Upload"])
        
        if input_method == "Manual Text Entry":
            col1, col2 = st.columns([2, 1])
            
            with col1:
                text_input = st.text_area("Enter text:", height=200, 
                                        placeholder="Enter text in any Indian language...")
                
            with col2:
                source = st.text_input("Source", placeholder="e.g., Wikipedia, News, Book")
                category = st.selectbox("Category", TEXT_CATEGORIES)
                contributor = st.text_input("Contributor Name", placeholder="Your name")
                
                # Manual language selection (optional)
                manual_lang = st.selectbox("Manual Language Selection (Optional)", 
                                         ["Auto-detect"] + list(INDIAN_LANGUAGES.keys()))
                manual_lang = None if manual_lang == "Auto-detect" else manual_lang
            
            if st.button("Add to Corpus", type="primary"):
                if text_input and source and contributor:
                    try:
                        lang_code, lang_name = corpus_manager.add_text_to_corpus(
                            text_input, source, category, contributor, manual_lang
                        )
                        st.success(f"✅ Text added successfully! Detected language: {lang_name}")
                        
                        # Show text preview
                        st.info(f"**Preview:** {text_input[:100]}...")
                        
                    except Exception as e:
                        st.error(f"Error adding text: {str(e)}")
                else:
                    st.warning("Please fill all required fields!")
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'csv'])
            
            if uploaded_file:
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    source = st.text_input("Source", placeholder="File name or origin")
                    category = st.selectbox("Category", TEXT_CATEGORIES)
                    contributor = st.text_input("Contributor Name")
                
                if st.button("Process File"):
                    if source and contributor:
                        try:
                            content = uploaded_file.read().decode('utf-8')
                            
                            # Split into sentences or paragraphs
                            texts = [t.strip() for t in content.split('\n') if t.strip() and len(t.strip()) > 10]
                            
                            if texts:
                                progress_bar = st.progress(0)
                                results = []
                                
                                for i, text in enumerate(texts):
                                    lang_code, lang_name = corpus_manager.add_text_to_corpus(
                                        text, source, category, contributor
                                    )
                                    results.append((lang_name, len(text.split())))
                                    progress_bar.progress((i + 1) / len(texts))
                                
                                st.success(f"✅ Processed {len(results)} texts from file!")
                                
                                # Show summary
                                if results:
                                    lang_summary = {}
                                    for lang, words in results:
                                        lang_summary[lang] = lang_summary.get(lang, 0) + words
                                    
                                    st.write("**Summary:**")
                                    for lang, word_count in lang_summary.items():
                                        st.write(f"- {lang}: {word_count} words")
                            else:
                                st.warning("No valid text found in the file!")
                                    
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                    else:
                        st.warning("Please provide source and contributor information!")
    
    elif page == "📊 Corpus Statistics":
        st.header("Corpus Statistics")
        
        stats = corpus_manager.get_corpus_stats()
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Texts", stats['total_texts'])
        col2.metric("Total Words", f"{stats['total_words']:,}")
        col3.metric("Total Characters", f"{stats['total_chars']:,}")
        col4.metric("Languages", len(stats['language_stats']))
        
        # Language distribution
        if not stats['language_stats'].empty:
            st.subheader("Language Distribution")
            
            if PLOTLY_AVAILABLE:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart for texts
                    fig_pie = px.pie(stats['language_stats'], 
                                   values='total_texts', 
                                   names='language_name',
                                   title="Distribution by Number of Texts")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart for words
                    fig_bar = px.bar(stats['language_stats'], 
                                   x='language_name', 
                                   y='total_words',
                                   title="Words per Language")
                    fig_bar.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                # Use simple charts
                st.write("**Texts per Language:**")
                chart_data = stats['language_stats'][['language_name', 'total_texts']]
                if not chart_data.empty:
                    st.bar_chart(chart_data.set_index('language_name'))
                
                st.write("**Words per Language:**")
                chart_data2 = stats['language_stats'][['language_name', 'total_words']]
                if not chart_data2.empty:
                    st.bar_chart(chart_data2.set_index('language_name'))
            
            # Language statistics table
            st.subheader("Detailed Language Statistics")
            st.dataframe(stats['language_stats'], use_container_width=True)
        
        # Category distribution
        if not stats['category_stats'].empty:
            st.subheader("Category Distribution")
            if PLOTLY_AVAILABLE:
                fig_category = px.bar(stats['category_stats'], 
                                    x='category', 
                                    y='count',
                                    title="Texts per Category")
                st.plotly_chart(fig_category, use_container_width=True)
            else:
                chart_data = stats['category_stats'][['category', 'count']]
                st.bar_chart(chart_data.set_index('category'))
    
    elif page == "🔍 Search Corpus":
        st.header("Search Corpus")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_query = st.text_input("Search Query", placeholder="Enter search terms...")
        
        with col2:
            stats = corpus_manager.get_corpus_stats()
            available_languages = ["All"]
            if not stats['language_stats'].empty:
                available_languages.extend(list(stats['language_stats']['language_code'].unique()))
            selected_language = st.selectbox("Filter by Language", available_languages)
        
        with col3:
            selected_category = st.selectbox("Filter by Category", ["All"] + TEXT_CATEGORIES)
        
        if st.button("Search") and search_query:
            results = corpus_manager.search_corpus(search_query, selected_language, selected_category)
            
            if not results.empty:
                st.success(f"Found {len(results)} results")
                
                for idx, row in results.iterrows():
                    with st.expander(f"{row['language_name']} - {row['category']} ({row['word_count']} words)"):
                        st.write(f"**Source:** {row['source']}")
                        st.write(f"**Date:** {row['date_added']}")
                        st.write(f"**Contributor:** {row['contributor']}")
                        st.write("**Text:**")
                        st.write(row['text'])
            else:
                st.warning("No results found!")
    
    elif page == "📥 Export Data":
        st.header("Export Corpus Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stats = corpus_manager.get_corpus_stats()
            available_languages = ["All"]
            if not stats['language_stats'].empty:
                available_languages.extend(list(stats['language_stats']['language_code'].unique()))
            export_language = st.selectbox("Select Language to Export", available_languages)
        
        with col2:
            export_format = st.selectbox("Export Format", ["CSV", "JSON"])
        
        if st.button("Generate Export"):
            try:
                if export_format == "CSV":
                    data = corpus_manager.export_corpus(export_language, 'csv')
                    st.download_button(
                        label="Download CSV",
                        data=data,
                        file_name=f"bhasha_seva_corpus_{export_language}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    data = corpus_manager.export_corpus(export_language, 'json')
                    st.download_button(
                        label="Download JSON",
                        data=data,
                        file_name=f"bhasha_seva_corpus_{export_language}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
                st.success("Export ready for download!")
                
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Bhasha Seva** 🇮🇳")
    st.sidebar.markdown("Preserving Indian Languages")
    
    # Installation instructions in sidebar
    with st.sidebar.expander("📦 Installation Guide"):
        st.markdown("""
        **To install missing packages:**
        
        ```bash
        pip install plotly
        pip install langdetect
        ```
        
        **Full installation:**
        ```bash
        pip install streamlit pandas plotly langdetect numpy sqlite3
        ```
        """)

if __name__ == "__main__":
    main()
