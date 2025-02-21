from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import unicodedata
import contractions
from enum import Enum

class NumberProcessor(Enum):
    NoAction = "no_action"
    ToString = "to_string"
    Remove = "remove"

class TextPreprocessHelper:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    """
    Purpose: 
        1. Perform basic cleanup of text, including removing HTML and brackets.
        2. Dimensionality reduction
    """
    def denoise_text(self, text):
        return self.__remove_brackets(self.__remove_html(text)).lower()
    
    def __remove_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def __remove_brackets(self, text):
        return re.sub(r'\[[^]]*\]', '', text)
    

    """
    Purpose: Expands abbreviations, e.g. "can't" → "cannot", "I'm" → "I am".
    """
    def expand_contractions(self, text):
        return contractions.fix(text)
    

    """
    Additional: support remove @someone and email address
    Purpose: improve generalization, dimensionality reduction
    ✅ Improve text consistency
    ✅ Enhance model robustness
    ✅ Reduce noise in data
    ✅ Optimize feature extraction
    ✅ Improve computational efficiency
    ✅ Facilitate better text clustering & classification

    🚀 Overall Goal: Standardize text for improved NLP performance.
    """
    def normalize(self, words, mentions_and_emails = True, non_ascii = True, lowercase = True, punctuation = True, stopwords = True, number = NumberProcessor.NoAction):
        if mentions_and_emails :
            words = self.__remove_mentions_and_emails(words)
        if non_ascii : 
            words = self.__remove_non_ascii(words)
        if lowercase:
            words = self.__to_lowercase(words)
        if punctuation :
            words = self.__remove_punctuation(words)
        if stopwords :
            words = self.__remove_stopwords(words)
        if number == NumberProcessor.ToString :
            words = self.__replace_numbers(words)
        elif number == NumberProcessor.Remove : 
            words = self.__remove_numbers(words)

        return words
    
    def __remove_mentions_and_emails(self, words):
        """Remove @someone and email address"""
        return [word for word in words if not word.startswith('@') and not re.match(r"[^@\s]+@[^@\s]+\.[a-zA-Z]+", word)]

    def __remove_non_ascii(self, words):
        return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8') for word in words]
    
    def __to_lowercase(words):
        return [word.lower() for word in words]

    def __remove_punctuation(self, words):
        return [re.sub(r'[^\w\s]', '', word) for word in words if word]

    def __remove_stopwords(self, words):
        return [word for word in words if word not in self.stop_words]

    def __replace_numbers(self, words):
        return [self.inflect_engine.number_to_words(word) if word.isdigit() else word for word in words]

    def __remove_numbers(self, words):
        return [word for word in words if not re.fullmatch(r"\d+(\.\d+)?", word)]