import re
import unicodedata
import spacy
import stanza
import inflect
import contractions
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from textblob import Word
from enum import Enum


class Tokenizer(Enum):
    NLTK = "nltk"
    SPACY = "spacy"
    STANZA = "stanza"
    REGEX = "regex"
    SPLIT = "split"

# Number处理策略
class NumberProcessor(Enum):
    NoAction = "no_action"
    ToString = "to_string"
    Remove = "remove"

# Stemming策略
class Stemmer(Enum):
    NoAction = "no_action"
    Porter = "porter"
    Snowball = "snowball"
    Lancaster = "lancaster"

# Lemmatization策略
class Lemmatizer(Enum):
    NoAction = "no_action"
    Wordnet = "wordnet"
    Spacy = "spacy"
    Textblob = "textblob"
    Stanza = "stanza"

"""
libraries:
pip install nltk spacy stanza textblob beautifulsoup4 contractions inflect
python -m spacy download en_core_web_sm

if use stanza, make sure executed below command:
import stanza
stanza.download("en")
"""


"""
## **完整的参数描述（包含必然执行的操作说明）**
> ✅ **表示该步骤是必然执行的**  
> 🔄 **表示该步骤可以根据参数配置进行执行或跳过**  

### **1. expand_contractions**
   - **Description:** Expands contractions, e.g., `"can't"` → `"cannot"`.  
   - **Possible values:** `True` (default) / `False`.  
   - **Execution:** 🔄（可选，取决于参数）  

### **2. remove_stopwords**
   - **Description:** Removes stopwords such as `"the"`, `"is"`, `"in"`.  
   - **Possible values:** `True` (default) / `False`.  
   - **Execution:** 🔄（可选，取决于参数）  

### **3. number_action**
   - **Description:** Defines how numbers should be handled.  
   - **Possible values:**  
     - `NumberProcessor.NoAction` (default): Keeps numbers as they are.  
     - `NumberProcessor.ToString`: Converts numbers to words, e.g., `"10"` → `"ten"`.  
     - `NumberProcessor.Remove`: Removes all numbers.  
   - **Execution:** 🔄（可选，取决于参数）  

### **4. stemming**
   - **Description:** Specifies whether stemming should be applied.  
   - **Possible values:**  
     - `Stemmer.NoAction` (default): No stemming applied.  
     - `Stemmer.Porter`: Uses **Porter Stemmer** (suitable for most NLP tasks).  
     - `Stemmer.Snowball`: Uses **Snowball Stemmer** (Porter2, more advanced).  
     - `Stemmer.Lancaster`: Uses **Lancaster Stemmer** (more aggressive and may over-truncate words).  
   - **Execution:** 🔄（可选，取决于参数）  

### **5. lemmatization**
   - **Description:** Defines whether lemmatization should be applied.  
   - **Possible values:**  
     - `Lemmatizer.NoAction` (default): No lemmatization applied.  
     - `Lemmatizer.Wordnet`: Uses **WordNet Lemmatizer** (dictionary-based lemmatization).  
     - `Lemmatizer.Spacy`: Uses **SpaCy Lemmatizer** (more precise, recommended).  
     - `Lemmatizer.Textblob`: Uses **TextBlob Lemmatizer** (simpler but less accurate).  
     - `Lemmatizer.Stanza`: Uses **Stanza Lemmatizer** (supports multiple languages).  
   - **Execution:** 🔄（可选，取决于参数）  
   - **Note:** **If `tokenization=Tokenizer.SPACY` or `tokenization=Tokenizer.STANZA`, lemmatization is automatically applied and does not need to be explicitly executed again.**  

### **6. tokenization**
   - **Description:** Defines how text is split into tokens (words or subwords).  
   - **Possible values:**  
     - `Tokenizer.NLTK` (default): Uses **NLTK’s `word_tokenize()`**. Suitable for general NLP tasks.  
     - `Tokenizer.SPACY`: Uses **SpaCy**. More advanced and includes lemmatization.  
     - `Tokenizer.STANZA`: Uses **Stanza**. Suitable for multilingual text, includes lemmatization.  
     - `Tokenizer.REGEX`: Uses **Regular Expressions** (`re.findall(r'\b\w+\b', text)`). Suitable for quick text preprocessing.  
     - `Tokenizer.SPLIT`: Uses **Python’s `.split()`**, simple space-based tokenization.  
   - **Execution:** ✅（必然执行）  
   - **Note:** If `Tokenizer.SPACY` or `Tokenizer.STANZA` is used, **Lemmatization is automatically included and will be skipped** in the later steps.  

---

## **必然执行的预处理步骤**
这些步骤 **不受参数控制，始终执行**：
1. **Remove HTML tags** (`__remove_html(text)`) ✅
2. **Remove brackets** (`__remove_brackets(text)`) ✅
3. **Convert text to lowercase** (`text.lower()`) ✅
4. **Tokenization** (`__tokenize(text)`) ✅
5. **Remove punctuation** (`__remove_punctuation(words)`) ✅
6. **Remove non-ASCII characters** (`__remove_non_ascii(words)`) ✅

"""

class TextPreprocessor:
    def __init__(self, expand_contractions=True, remove_stopwords=True,
                 number_action=NumberProcessor.NoAction,
                 stemming=Stemmer.NoAction, lemmatization=Lemmatizer.NoAction,
                 tokenization=Tokenizer.NLTK):

        self.expand_contractions = expand_contractions
        self.tokenization_method = tokenization  # 新增 tokenization 选项

        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()

        # 设置数字处理方式
        if number_action == NumberProcessor.ToString:
            self.inflect_engine = inflect.engine()
            self.number_processor = self.__replace_numbers
        elif number_action == NumberProcessor.Remove:
            self.number_processor = self.__remove_numbers
        else:
            self.number_processor = self.__NO_ACTION

        # 设置 Stemming
        if stemming == Stemmer.Porter:
            self.porter_stemmer = PorterStemmer()
            self.stemmer = self.__stem_with_porter
        elif stemming == Stemmer.Snowball:
            self.snowball_stemmer = SnowballStemmer("english")
            self.stemmer = self.__stem_with_snowball
        elif stemming == Stemmer.Lancaster:
            self.lancaster_stemmer = LancasterStemmer()
            self.stemmer = self.__stem_with_lancaster
        else:
            self.stemmer = self.__NO_ACTION

        # 设置 Lemmatization
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
        else:
            self.lemmatizer = self.__NO_ACTION

    def process(self, text):
        # Step 1: Remove HTML
        text = self.__remove_html(text)
        # Step 2: Remove brackets
        text = self.__remove_brackets(text)
        # Step 3: Expand contractions
        if self.expand_contractions:
            text = self.__expand_contractions(text)
        # Step 4: Convert to lowercase
        text = text.lower()

        # Step 5: Tokenization
        words = self.__tokenize(text)

        # Step 6: Remove punctuation
        words = self.__remove_punctuation(words)
        # Step 7: Remove non-ASCII characters
        words = self.__remove_non_ascii(words)
        # Step 8: Process numbers
        words = self.number_processor(words)
        # Step 9: Remove stopwords
        words = self.__remove_stopwords(words)
        # Step 10: Apply stemming
        words = self.stemmer(words)
        # **Step 11: 仅在非 SpaCy 或 Stanza 选项下进行 Lemmatization**
        if self.tokenization_method not in [Tokenizer.SPACY, Tokenizer.STANZA]:
            words = self.lemmatizer(words)
        return words

    def __tokenize(self, text):
        """根据选择的 tokenization 方法对文本进行分词"""
        if self.tokenization_method == Tokenizer.NLTK:
            return word_tokenize(text)
        elif self.tokenization_method == Tokenizer.SPACY:
            doc = self.spacy_nlp(text)
            return [token.text for token in doc]
        elif self.tokenization_method == Tokenizer.STANZA:
            doc = self.stanza_nlp(text)
            return [word.text for sent in doc.sentences for word in sent.words]
        elif self.tokenization_method == Tokenizer.REGEX:
            return re.findall(r'\b\w+\b', text)
        elif self.tokenization_method == Tokenizer.SPLIT:
            return text.split()
        else:
            raise ValueError("Unsupported tokenization method!")

    def __remove_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def __remove_brackets(self, text):
        return re.sub(r'\[[^]]*\]', '', text)

    def __expand_contractions(self, text):
        return contractions.fix(text)

    def __remove_non_ascii(self, words):
        return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8') for word in words]

    def __remove_punctuation(self, words):
        return [re.sub(r'[^\w\s]', '', word) for word in words if word]

    def __remove_stopwords(self, words):
        return [word for word in words if word not in self.stop_words]

    def __replace_numbers(self, words):
        return [self.inflect_engine.number_to_words(word) if word.isdigit() else word for word in words]

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
    
"""
preprocessor = TextPreprocessor(
    expand_contractions=True, 
    remove_stopwords=False,  # 不去除停用词，保留更多信息
    number_action=NumberProcessor.NoAction,  # 保留数字
    stemming=Stemmer.NoAction,  # 不进行 Stemming
    lemmatization=Lemmatizer.NoAction,  # 不进行 Lemmatization
    tokenization=Tokenizer.NLTK  # 使用 NLTK 进行分词
)

preprocessor = TextPreprocessor(
    expand_contractions=True, 
    remove_stopwords=True,  # 去除停用词，提高分类模型性能
    number_action=NumberProcessor.Remove,  # 移除所有数字，避免噪声
    stemming=Stemmer.Porter,  # Porter Stemmer，减少单词变形
    lemmatization=Lemmatizer.NoAction,  # 使用 Stemming 而非 Lemmatization
    tokenization=Tokenizer.NLTK  # 使用 NLTK 进行分词
)

preprocessor = TextPreprocessor(
    expand_contractions=True, 
    remove_stopwords=True,  # 去除停用词，提高分类模型性能
    number_action=NumberProcessor.Remove,  # 移除所有数字，避免噪声
    stemming=Stemmer.Porter,  # Porter Stemmer，减少单词变形
    lemmatization=Lemmatizer.NoAction,  # 使用 Stemming 而非 Lemmatization
    tokenization=Tokenizer.NLTK  # 使用 NLTK 进行分词
)

preprocessor = TextPreprocessor(
    expand_contractions=True, 
    remove_stopwords=False,  # 让 Transformer 模型自行学习停用词影响
    number_action=NumberProcessor.NoAction,  # 保留原始数字信息
    stemming=Stemmer.NoAction,  # 不使用 Stemming，保持单词完整性
    lemmatization=Lemmatizer.Spacy,  # SpaCy 进行高质量 Lemmatization
    tokenization=Tokenizer.SPACY  # 使用 SpaCy 进行分词（自带 lemmatization）
)

preprocessor = TextPreprocessor(
    expand_contractions=True, 
    remove_stopwords=True,  # 去除停用词，提高主题建模质量
    number_action=NumberProcessor.Remove,  # 移除数字，避免数字影响主题
    stemming=Stemmer.NoAction,  # 主题建模通常不需要 Stemming
    lemmatization=Lemmatizer.Wordnet,  # 使用 WordNet 进行 Lemmatization
    tokenization=Tokenizer.NLTK  # 使用 NLTK 进行分词
)


preprocessor = TextPreprocessor(
    expand_contractions=True, 
    remove_stopwords=False,  # 保留停用词，依存分析需要完整句子结构
    number_action=NumberProcessor.NoAction,  # 保留原始数字信息
    stemming=Stemmer.NoAction,  # 不进行 Stemming，保持句法结构
    lemmatization=Lemmatizer.Stanza,  # Stanza 进行高级 Lemmatization
    tokenization=Tokenizer.STANZA  # 使用 Stanza 进行分词（自带语法分析）
)

"""