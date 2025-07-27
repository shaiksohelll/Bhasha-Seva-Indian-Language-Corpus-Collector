"""
Corpus Manager - Core functionality for managing the Indian language corpus
"""
import sqlite3
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from config import (
    DATABASE_PATH, INDIAN_LANGUAGES, TEXT_CATEGORIES, 
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, DB_VERSION,
    ERROR_MESSAGES, SUCCESS_MESSAGES
)
from src.language_detector import LanguageDetector
from src.utils.validators import TextValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorpusManager:
    """Main class for managing the corpus database and operations"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_PATH
        self.language_detector = LanguageDetector()
        self.validator = TextValidator()
        self.init_database()
    
    def init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create corpus_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corpus_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    language_code TEXT NOT NULL,
                    language_name TEXT NOT NULL,
                    source TEXT,
                    category TEXT,
                    contributor TEXT,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    word_count INTEGER,
                    char_count INTEGER,
                    sentences_count INTEGER,
                    confidence_score REAL,
                    metadata TEXT  -- JSON field for additional data
                )
            ''')
            
            # Create language_stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS language_stats (
                    language_code TEXT PRIMARY KEY,
                    language_name TEXT,
                    total_texts INTEGER DEFAULT 0,
                    total_words INTEGER DEFAULT 0,
                    total_chars INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create contributors table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contributors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    total_contributions INTEGER DEFAULT 0,
                    total_words_contributed INTEGER DEFAULT 0,
                    first_contribution TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_contribution TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create database metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert or update database version
            cursor.execute('''
                INSERT OR REPLACE INTO db_metadata (key, value)
                VALUES ('version', ?)
            ''', (DB_VERSION,))
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_language_code ON corpus_data(language_code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON corpus_data(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_contributor ON corpus_data(contributor)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date_added ON corpus_data(date_added)')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def add_text_to_corpus(self, 
                          text: str, 
                          source: str = None, 
                          category: str = None, 
                          contributor: str = None,
                          manual_language: str = None,
                          metadata: Dict = None) -> Tuple[str, str, float]:
        """
        Add text to corpus database
        
        Args:
            text: The text content
            source: Source of the text
            category: Category/type of text
            contributor: Name of contributor
            manual_language: Manually specified language code
            metadata: Additional metadata as dictionary
            
        Returns:
            Tuple of (language_code, language_name, confidence_score)
        """
        try:
            # Validate input text
            validation_result = self.validator.validate_text(text)
            if not validation_result['is_valid']:
                raise ValueError(validation_result['error'])
            
            # Detect or use manual language
            if manual_language and manual_language in INDIAN_LANGUAGES:
                lang_code = manual_language
                lang_name = INDIAN_LANGUAGES[manual_language]
                confidence = 1.0  # Manual selection is 100% confident
            else:
                detection_result = self.language_detector.detect_language(text)
                lang_code = detection_result['language_code']
                lang_name = detection_result['language_name'] 
                confidence = detection_result['confidence']
            
            # Calculate text statistics
            word_count = len(text.split())
            char_count = len(text)
            sentences_count = text.count('.') + text.count('!') + text.count('?')
            
            # Prepare metadata
            metadata_json = json.dumps(metadata or {})
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO corpus_data 
                (text, language_code, language_name, source, category, contributor, 
                 word_count, char_count, sentences_count, confidence_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (text, lang_code, lang_name, source or 'Unknown', 
                  category or 'Other', contributor or 'Anonymous',
                  word_count, char_count, sentences_count, confidence, metadata_json))
            
            # Update language statistics
            self._update_language_stats(cursor, lang_code, lang_name, word_count, char_count, confidence)
            
            # Update contributor statistics
            self._update_contributor_stats(cursor, contributor or 'Anonymous', word_count)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added text to corpus: {lang_name} ({word_count} words)")
            return lang_code, lang_name, confidence
            
        except Exception as e:
            logger.error(f"Failed to add text to corpus: {e}")
            raise
    
    def _update_language_stats(self, cursor, lang_code: str, lang_name: str, 
                             word_count: int, char_count: int, confidence: float) -> None:
        """Update language statistics in database"""
        cursor.execute('''
            INSERT OR REPLACE INTO language_stats 
            (language_code, language_name, total_texts, total_words, total_chars, avg_confidence)
            VALUES (?, ?, 
                COALESCE((SELECT total_texts FROM language_stats WHERE language_code = ?), 0) + 1,
                COALESCE((SELECT total_words FROM language_stats WHERE language_code = ?), 0) + ?,
                COALESCE((SELECT total_chars FROM language_stats WHERE language_code = ?), 0) + ?,
                (COALESCE((SELECT avg_confidence * total_texts FROM language_stats WHERE language_code = ?), 0) + ?) / 
                (COALESCE((SELECT total_texts FROM language_stats WHERE language_code = ?), 0) + 1)
            )
        ''', (lang_code, lang_name, lang_code, lang_code, word_count, 
              lang_code, char_count, lang_code, confidence, lang_code))
    
    def _update_contributor_stats(self, cursor, contributor: str, word_count: int) -> None:
        """Update contributor statistics in database"""
        cursor.execute('''
            INSERT OR REPLACE INTO contributors 
            (name, total_contributions, total_words_contributed, first_contribution, last_contribution)
            VALUES (?, 
                COALESCE((SELECT total_contributions FROM contributors WHERE name = ?), 0) + 1,
                COALESCE((SELECT total_words_contributed FROM contributors WHERE name = ?), 0) + ?,
                COALESCE((SELECT first_contribution FROM contributors WHERE name = ?), CURRENT_TIMESTAMP),
                CURRENT_TIMESTAMP
            )
        ''', (contributor, contributor, contributor, word_count, contributor))
    
    def get_corpus_stats(self) -> Dict:
        """Get comprehensive corpus statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Overall statistics
            overall_stats = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total_texts,
                    SUM(word_count) as total_words,
                    SUM(char_count) as total_chars,
                    AVG(word_count) as avg_words_per_text,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(DISTINCT language_code) as unique_languages,
                    COUNT(DISTINCT contributor) as unique_contributors
                FROM corpus_data
            ''', conn).iloc[0].to_dict()
            
            # Language-wise statistics
            language_stats = pd.read_sql_query('''
                SELECT * FROM language_stats 
                ORDER BY total_words DESC
            ''', conn)
            
            # Category-wise statistics
            category_stats = pd.read_sql_query('''
                SELECT 
                    category, 
                    COUNT(*) as count, 
                    SUM(word_count) as total_words,
                    AVG(word_count) as avg_words,
                    AVG(confidence_score) as avg_confidence
                FROM corpus_data 
                GROUP BY category 
                ORDER BY count DESC
            ''', conn)
            
            # Contributor statistics
            contributor_stats = pd.read_sql_query('''
                SELECT * FROM contributors 
                ORDER BY total_contributions DESC
                LIMIT 10
            ''', conn)
            
            # Recent additions
            recent_additions = pd.read_sql_query('''
                SELECT language_name, category, word_count, date_added, contributor
                FROM corpus_data 
                ORDER BY date_added DESC 
                LIMIT 10
            ''', conn)
            
            # Daily statistics for the last 30 days
            daily_stats = pd.read_sql_query('''
                SELECT 
                    DATE(date_added) as date,
                    COUNT(*) as texts_added,
                    SUM(word_count) as words_added
                FROM corpus_data 
                WHERE date_added >= datetime('now', '-30 days')
                GROUP BY DATE(date_added)
                ORDER BY date DESC
            ''', conn)
            
            conn.close()
            
            return {
                'overall': overall_stats,
                'languages': language_stats,
                'categories': category_stats,
                'contributors': contributor_stats,
                'recent_additions': recent_additions,
                'daily_stats': daily_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get corpus statistics: {e}")
            raise
    
    def search_corpus(self, 
                     query: str, 
                     language: str = None, 
                     category: str = None,
                     contributor: str = None,
                     limit: int = 100) -> pd.DataFrame:
        """
        Search corpus data with various filters
        
        Args:
            query: Search query string
            language: Language code filter
            category: Category filter  
            contributor: Contributor filter
            limit: Maximum number of results
            
        Returns:
            DataFrame with search results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build dynamic query
            sql = """
                SELECT id, text, language_name, category, source, contributor, 
                       word_count, char_count, confidence_score, date_added
                FROM corpus_data 
                WHERE text LIKE ?
            """
            params = [f"%{query}%"]
            
            if language and language != "All":
                sql += " AND language_code = ?"
                params.append(language)
            
            if category and category != "All":
                sql += " AND category = ?"
                params.append(category)
                
            if contributor and contributor != "All":
                sql += " AND contributor = ?"
                params.append(contributor)
            
            sql += " ORDER BY confidence_score DESC, date_added DESC LIMIT ?"
            params.append(limit)
            
            results = pd.read_sql_query(sql, conn, params=params)
            conn.close()
            
            logger.info(f"Search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def export_corpus(self, 
                     language: str = None, 
                     category: str = None,
                     format: str = 'csv',
                     limit: int = None) -> str:
        """
        Export corpus data in specified format
        
        Args:
            language: Language code filter
            category: Category filter
            format: Export format ('csv', 'json', 'txt')
            limit: Maximum number of records
            
        Returns:
            Exported data as string
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query
            sql = "SELECT * FROM corpus_data"
            params = []
            conditions = []
            
            if language and language != "All":
                conditions.append("language_code = ?")
                params.append(language)
            
            if category and category != "All":
                conditions.append("category = ?")
                params.append(category)
            
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            
            sql += " ORDER BY date_added DESC"
            
            if limit:
                sql += f" LIMIT {limit}"
            
            df = pd.read_sql_query(sql, conn, params=params)
            conn.close()
            
            # Format output
            if format.lower() == 'csv':
                return df.to_csv(index=False)
            elif format.lower() == 'json':
                return df.to_json(orient='records', indent=2, ensure_ascii=False)
            elif format.lower() == 'txt':
