from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import unicodedata
import contractions
import inflect
from enum import Enum

class NumberProcessor(Enum):
    NoAction = "no_action"
    ToString = "to_string"
    Remove = "remove"

class TextPreprocessHelper:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.inflect_engine = inflect.engine()
    
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
    
    def remove_mentions_and_emails(self, text):
        """Remove @someone and email addresses from text"""
        text = re.sub(r'@\w+', '', text) # remove @username
        text = re.sub(r"[^@\s]+@[^@\s]+\.[a-zA-Z]+", '', text) # remove Email address
        return text
    
    def remove_non_ascii_in_text(self, text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    def remove_non_ascii(self, words):
        return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8') for word in words]

    def remove_punctuation(self, words):
        return [re.sub(r'[^\w\s]', '', word) for word in words if word]

    # def replace_numbers(self, words):
        # return [self.inflect_engine.number_to_words(word) if word.isdigit() else word for word in words]

    def replace_numbers(self, words):
        new_words = []
        for word in words:
            if word.isdigit():  # 只处理纯数字
                try:
                    word = self.inflect_engine.number_to_words(word)
                except:
                    pass  # 如果转换失败，就跳过
            new_words.append(word)
        return new_words

    def remove_numbers(self, words):
        return [word for word in words if not re.fullmatch(r"\d+(\.\d+)?", word)]