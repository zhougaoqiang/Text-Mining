import unicodedata
import contractions  
import inflect       
import re
import spacy
import stanza
from textblob import Word
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from enum import IntEnum

class NumberProcessor(IntEnum):
    NoAction = 1
    ToString = 2
    Remove = 3

class Stemmer(IntEnum):
    NoAction = 1
    Porter = 2
    Snowball = 3
    Lancaster = 4

class Lemmatizer(IntEnum):
    NoAction = 1
    Wordnet = 2
    Spacy = 3
    Textblob = 4
    Stanza = 5

"""
libraries:
pip install nltk spacy stanza textblob beautifulsoup4 contractions inflect
python -m spacy download en_core_web_sm

if use stanza, make sure executed below command:
import stanza
stanza.download("en")
"""


"""
Parameter Description
    1. expand_contractions:
        ◦ Description: Expands contractions, e.g., "can't" → "cannot". 
        ◦ Possible values: True (default) / False. 
    2. remove_stopwords:
        ◦ Description: Removes stopwords such as "the", "is", "in". 
        ◦ Possible values: True (default) / False. 
    3. number_action:
        ◦ Description: Defines how numbers should be handled. 
        ◦ Possible values: 
            ▪ NumberProcessor.NoAction (default): Keeps numbers as they are. 
            ▪ NumberProcessor.ToString: Converts numbers to words, e.g., "10" → "ten". 
            ▪ NumberProcessor.Remove: Removes all numbers. 
    4. stemming:
        ◦ Description: Specifies whether stemming should be applied. 
        ◦ Possible values: 
            ▪ Stemmer.NoAction (default): No stemming applied. 
            ▪ Stemmer.Porter: Uses Porter Stemmer (suitable for most NLP tasks). 
            ▪ Stemmer.Snowball: Uses Snowball Stemmer (Porter2, more advanced). 
            ▪ Stemmer.Lancaster: Uses Lancaster Stemmer (more aggressive and may over-truncate words). 
    5. lemmatization:
        ◦ Description: Defines whether lemmatization should be applied. 
        ◦ Possible values: 
            ▪ Lemmatizer.NoAction (default): No lemmatization applied. 
            ▪ Lemmatizer.Wordnet: Uses WordNet Lemmatizer (dictionary-based lemmatization). 
            ▪ Lemmatizer.Spacy: Uses Spacy Lemmatizer (more precise, recommended). 
            ▪ Lemmatizer.Textblob: Uses TextBlob Lemmatizer (simpler but less accurate). 
            ▪ Lemmatizer.Stanza: Uses Stanza Lemmatizer (supports multiple languages). 

stemming and lemmatization can work together, but recommend using Lemmatization (Spacy or WordNet)            
"""

class TextPreprocessor:
    def __init__(self, expand_contractions=True,remove_stopwords=True,
                 number_action = NumberProcessor.NoAction, 
                 stemming=Stemmer.NoAction, lemmatization=Lemmatizer.NoAction):
        
        self.expand_contractions = expand_contractions

        if remove_stopwords :
            self.stop_words = set(stopwords.words('english'))
        else :
            self.stop_words = set()

        if number_action == NumberProcessor.ToString:
            self.inflect_engine = inflect.engine()
            self.number_processor = self.__replace_numbers
        elif number_action == NumberProcessor.Remove:
            self.number_processor = self.__remove_numbers
        else:  #NumberProcessor.NoActon
            self.number_processor = self.__NO_ACTION
            
        # set Stemming
        if stemming == Stemmer.Porter:
            self.stemmer = self.__stem_with_porter
            self.porter_stemmer = PorterStemmer()
        elif stemming == Stemmer.Snowball:
            self.snowball_stemmer = SnowballStemmer("english")
            self.stemmer = self.__stem_with_snowball
        elif stemming == Stemmer.Lancaster:
            self.lancaster_stemmer = LancasterStemmer()
            self.stemmer = self.__stem_with_lancaster
        else:  # NoAction
            self.stemmer = self.__NO_ACTION

        # set Lemmatization
        if lemmatization == Lemmatizer.Wordnet:
            self.wordnet_lemmatizer = WordNetLemmatizer()
            self.lemmatizer = self.__lemmatize_with_wordnet
        elif lemmatization == Lemmatizer.Spacy:
            self.spacy_nlp = spacy.load("en_core_web_sm")
            self.lemmatizer = self.__lemmatize_with_spacy
        elif lemmatization == Lemmatizer.Textblob:
            self.lemmatizer = self.__lemmatize_with_textblob
        elif lemmatization == Lemmatizer.Stanza:
            self.stanza_nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos,lemma")
            self.lemmatizer = self.__lemmatize_with_stanza
        else:  # NoAction
            self.lemmatizer = self.__NO_ACTION

    ######public functions
    def process(self, text) :
        # Step 1: Remove HTML (always exec)
        text = self.__remove_html(text)
        # Step 2: Remove brackets (always exec)
        text = self.__remove_brackets(text)
        # Step 3: Expand contractions
        if self.expand_contractions:
            text = self.__expand_contractions(text)
        # Step 4: Remove non-ASCII characters (always exec)
        text = self.__remove_non_ascii(text) 
        # Step 5: Convert to lowercase (always exec)
        text = self.__to_lowercase(text)
        # Step 6: Remove punctuation (always exec)
        text = self.__remove_punctuation(text)
        # Step 7: Process numbers
        text = self.number_processor(text)
        # Step 8: Remove stopwords
        text = self.__remove_stopwords(text)
        # Step 9: Apply stemming
        text = self.stemmer(text)
        # Step 10: Apply lemmatization
        text = self.lemmatizer(text)
        return text

    #####private functions
    def __remove_html(self, text) :
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    
    def __remove_brackets(self, text):
        return re.sub(r'\[[^]]*\]', '', text)

    def __remove_non_ascii(self, words):
        return [ unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8') for word in words]
    
    def __to_lowercase(self, words):
        return [word.lower() for word in words]
    
    def __remove_punctuation(self, words):
        return [new_word for new_word in (re.sub(r'[^\w\s]', '', word) for word in words) if new_word]
    
    def __expand_contractions(self, text) :
        return contractions.fix(text)
    
    def __remove_stopwords(self, words) :
        if not self.stop_words:
            return words
        return [word for word in words if word not in self.stop_words]

    def __replace_numbers(self, words):
        return [self.inflect_engine.number_to_words(word) 
                if word.isdigit() else word for word in words]

    def __remove_numbers(self, words):
        return [word for word in words if not re.fullmatch(r"\d+(\.\d+)?", word)]
    
    def __NO_ACTION(self, words):
        return words

     # Stemming methods
    def __stem_with_porter(self, words):
        return [self.porter_stemmer.stem(word) for word in words]

    def __stem_with_snowball(self, words):
        return [self.snowball_stemmer.stem(word) for word in words]

    def __stem_with_lancaster(self, words):
        return [self.lancaster_stemmer.stem(word) for word in words]

    # Lemmatization methods
    def __lemmatize_with_wordnet(self, words):
        return [self.wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

    def __lemmatize_with_spacy(self, words):
        doc = self.spacy_nlp(" ".join(words))
        return [token.lemma_ for token in doc]

    def __lemmatize_with_textblob(self, words):
        return [Word(word).lemmatize() for word in words]

    def __lemmatize_with_stanza(self, words):
        doc = self.stanza_nlp(" ".join(words))
        return [word.lemma for sent in doc.sentences for word in sent.words]