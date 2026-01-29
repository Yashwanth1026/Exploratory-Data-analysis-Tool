import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Setup NLTK (should handle downloading in main app or check availability)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    NLTK_AVAILABLE = True
except LookupError:
    NLTK_AVAILABLE = False
    # You might want to trigger download here or assume it's done elsewhere
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        NLTK_AVAILABLE = True
    except:
        NLTK_AVAILABLE = False

def text_preprocessing(df, text_col, steps):
    """Comprehensive text preprocessing"""
    if not NLTK_AVAILABLE:
        return df, ["NLTK not available"]
    
    df_copy = df.copy()
    
    if text_col not in df_copy.columns:
        return df_copy, []
    
    # Convert to string
    df_copy[text_col] = df_copy[text_col].astype(str)
    
    processed_steps = []
    
    try:
        if 'lowercase' in steps:
            df_copy[text_col] = df_copy[text_col].str.lower()
            processed_steps.append('lowercase')
        
        if 'remove_punct' in steps:
            df_copy[text_col] = df_copy[text_col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
            processed_steps.append('remove_punct')
        
        if 'remove_stopwords' in steps:
            stop_words = set(stopwords.words('english'))
            df_copy[text_col] = df_copy[text_col].apply(
                lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words])
            )
            processed_steps.append('remove_stopwords')
        
        if 'stemming' in steps:
            ps = PorterStemmer()
            df_copy[text_col] = df_copy[text_col].apply(
                lambda x: ' '.join([ps.stem(word) for word in word_tokenize(x)])
            )
            processed_steps.append('stemming')
        
        if 'lemmatization' in steps:
            lemmatizer = WordNetLemmatizer()
            df_copy[text_col] = df_copy[text_col].apply(
                lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)])
            )
            processed_steps.append('lemmatization')
        
        if 'word_count' in steps:
            df_copy[f'{text_col}_word_count'] = df_copy[text_col].apply(
                lambda x: len(word_tokenize(x))
            )
            processed_steps.append('word_count')
        
        if 'char_count' in steps:
            df_copy[f'{text_col}_char_count'] = df_copy[text_col].apply(len)
            processed_steps.append('char_count')
        
        if 'remove_numbers' in steps:
            df_copy[text_col] = df_copy[text_col].apply(lambda x: re.sub(r'\d+', '', x))
            processed_steps.append('remove_numbers')
        
        if 'remove_whitespace' in steps:
            df_copy[text_col] = df_copy[text_col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            processed_steps.append('remove_whitespace')
            
    except Exception as e:
        return df_copy, [str(e)]
    
    return df_copy, processed_steps
