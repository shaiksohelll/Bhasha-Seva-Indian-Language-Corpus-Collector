"""
Configuration settings for Bhasha Seva - Indian Language Corpus Collector
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATABASE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "exports").mkdir(exist_ok=True)
(DATA_DIR / "uploads").mkdir(exist_ok=True)
(DATA_DIR / "sample_data").mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database Configuration
DATABASE_PATH = DATABASE_DIR / "bhasha_seva_corpus.db"
BACKUP_PATH = DATA_DIR / "backups"

# Supported Indian Languages with their codes
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
    'mai': 'Maithili (मैथिली)',
    'bh': 'Bhojpuri (भोजपुरी)',
    'ne': 'Nepali (नेपाली)',
    'sa': 'Sanskrit (संस्कृत)',
    'kok': 'Konkani (कोंकणी)',
    'sd': 'Sindhi (سنڌي)',
    'ks': 'Kashmiri (کٲشُر)',
    'doi': 'Dogri (डोगरी)'
}

# Text Categories
TEXT_CATEGORIES = [
    "News & Journalism",
    "Literature & Poetry", 
    "Social Media",
    "Government Documents",
    "Academic Papers",
    "Religious Texts",
    "Folk Tales & Stories",
    "Technical Documentation",
    "Legal Documents",
    "Educational Content",
    "Historical Texts",
    "Scientific Articles",
    "Cultural Heritage",
    "Other"
]

# File Upload Settings
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_EXTENSIONS = {'.txt', '.csv', '.json', '.tsv'}
MAX_TEXT_LENGTH = 50000  # Maximum characters per text entry
MIN_TEXT_LENGTH = 10     # Minimum characters per text entry

# Language Detection Settings
DETECTION_CONFIDENCE_THRESHOLD = 0.7
DETECTION_SEED = 0  # For consistent results

# Export Settings
EXPORT_BATCH_SIZE = 1000
MAX_EXPORT_RECORDS = 10000

# Analytics Settings
DASHBOARD_REFRESH_INTERVAL = 300  # seconds
MAX_SEARCH_RESULTS = 100

# API Configuration (Optional - for future enhancements)
GOOGLE_TRANSLATE_API_KEY = os.getenv('GOOGLE_TRANSLATE_API_KEY')
AZURE_TEXT_ANALYTICS_KEY = os.getenv('AZURE_TEXT_ANALYTICS_KEY')
AZURE_TEXT_ANALYTICS_ENDPOINT = os.getenv('AZURE_TEXT_ANALYTICS_ENDPOINT')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = LOGS_DIR / "bhasha_seva.log"

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'Bhasha Seva - Indian Language Corpus Collector',
    'page_icon': '🇮🇳',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color Scheme
COLORS = {
    'primary': '#FF6B35',
    'secondary': '#004E89', 
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#3498DB',
    'light': '#F8F9FA',
    'dark': '#343A40'
}

# Language-specific colors for visualizations
LANGUAGE_COLORS = {
    'hi': '#FF6B35',  # Orange - Hindi
    'bn': '#28A745',  # Green - Bengali
    'te': '#007BFF',  # Blue - Telugu
    'mr': '#6F42C1',  # Purple - Marathi
    'ta': '#DC3545',  # Red - Tamil
    'ur': '#20C997',  # Teal - Urdu
    'gu': '#FD7E14',  # Orange - Gujarati
    'kn': '#6610F2',  # Indigo - Kannada
    'ml': '#E83E8C',  # Pink - Malayalam
    'pa': '#17A2B8',  # Info - Punjabi
}

# Database Schema Version
DB_VERSION = "1.0.0"

# Feature Flags
FEATURES = {
    'web_scraping': False,
    'batch_processing': True,
    'advanced_analytics': True,
    'user_authentication': False,
    'api_endpoints': False,
    'real_time_collaboration': False
}

# Performance Settings
PAGINATION_SIZE = 50
CACHE_TTL = 3600  # seconds
MAX_CONCURRENT_UPLOADS = 3

# Validation Rules
VALIDATION_RULES = {
    'contributor_name_max_length': 100,
    'source_max_length': 200,
    'category_required': True,
    'min_word_count': 3,
    'max_word_count': 10000
}

# Sample Data Configuration
SAMPLE_DATA_CONFIG = {
    'generate_samples': True,
    'sample_size_per_language': 10,
    'sample_text_length_range': (100, 500)
}

# Backup Configuration
BACKUP_CONFIG = {
    'auto_backup': True,
    'backup_interval_hours': 24,
    'max_backup_files': 7,
    'compress_backups': True
}

# Error Messages
ERROR_MESSAGES = {
    'db_connection': "Database connection failed. Please check the database file.",
    'invalid_language': "Language not supported or detection failed.",
    'text_too_short': f"Text must be at least {MIN_TEXT_LENGTH} characters long.",
    'text_too_long': f"Text cannot exceed {MAX_TEXT_LENGTH} characters.",
    'file_too_large': f"File size cannot exceed {MAX_UPLOAD_SIZE // (1024*1024)}MB.",
    'invalid_file_type': f"Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed.",
    'empty_required_field': "Please fill all required fields.",
    'search_failed': "Search operation failed. Please try again.",
    'export_failed': "Export operation failed. Please try again."
}

# Success Messages
SUCCESS_MESSAGES = {
    'text_added': "✅ Text successfully added to corpus!",
    'file_processed': "✅ File processed successfully!",
    'export_ready': "✅ Export file is ready for download!",
    'backup_created': "✅ Backup created successfully!",
    'data_imported': "✅ Data imported successfully!"
}

# Default Values
DEFAULTS = {
    'contributor': 'Anonymous',
    'source': 'Unknown',
    'category': 'Other',
    'language': 'auto-detect',
    'export_format': 'csv'
}
