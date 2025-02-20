
import unicodedata
import contractions  
import inflect       
import re
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


"""
strip_html(text)
作用：去除文本中的 HTML 标签。
原理：使用 BeautifulSoup 提取纯文本，去掉 HTML 代码。
"""
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

"""
remove_between_square_brackets(text)
作用：删除方括号 [...] 内的内容，通常用于去除引用或元数据。
原理：使用正则表达式 re.sub(r'\[[^]]*\]', '', text) 识别并移除方括号中的内容。
"""
def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

"""
denoise_text(text)
作用：对文本进行基础清理，包括去除 HTML 和方括号内容。
原理：调用 strip_html 和 remove_between_square_brackets 两个函数。
"""
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

"""
replace_contractions(text)
作用：扩展缩写词，例如 "can't" → "cannot"， "I'm" → "I am"。
原理：使用 contractions.fix() 自动替换英语缩写。
"""
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

"""
remove_non_ascii(words)
作用：移除非 ASCII 字符，保留标准的英文字符。
原理：使用 unicodedata.normalize('NFKD', word).encode('ascii', 'ignore') 进行字符转换，删除特殊符号。
"""
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


"""
"""
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


"""
remove_punctuation(words)
作用：去除标点符号，仅保留单词和空格。
原理：使用正则表达式 re.sub(r'[^\w\s]', '', word) 移除非字母数字字符。
"""
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


"""
replace_numbers(words)
作用：将数字转换为英文单词，例如 "123" → "one hundred twenty-three"。
原理：使用 inflect 库的 number_to_words() 方法进行数字转文本转换。
"""
def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

"""
将所有数字包括小数都移除
"""
def remove_numbers(words):
    """Remove all numeric occurrences (integers and floats) from a list of tokenized words"""
    new_words = []
    for word in words:
        # Check if the word is a number (integer or float)
        if not re.fullmatch(r"\d+(\.\d+)?", word):  # Matches integers (123) and floats (12.34)
            new_words.append(word)
    return new_words


"""
remove_stopwords(words)
作用：删除常见的英语停用词（stopwords），如 "the", "is", "in"，以减少文本噪音。
原理：使用 NLTK 的 stopwords.words('english') 进行匹配并过滤。
"""
def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

"""
stem_words(words)
作用：将单词简化为词根，例如 "running" → "run"， "better" → "bet"。
原理：使用 NLTK 的 LancasterStemmer 进行词干提取（Stemming）。
"""
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

"""
Porter Stemmer（最常用）
Snowball Stemmer（也叫 Porter2 Stemmer，更强大）
Lancaster Stemmer（你当前的方式，更激进）
"""
def stem_with_snowball(words):
    """Stem words using Snowball Stemmer"""
    stemmer = SnowballStemmer("english")  # 还支持 "spanish", "french" 等
    return [stemmer.stem(word) for word in words]

def stem_with_porter(words):
    """Stem words using Porter Stemmer"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def stem_with_lancaster(words):
    """Stem words using Lancaster Stemmer"""
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in words]

"""
lemmatize_verbs(words)
作用：将单词转换为其字典中的基本形式，同时保留原始意义，例如 "running" → "run"， "was" → "be"。
原理：使用 NLTK 的 WordNetLemmatizer 进行词形还原（Lemmatization）。
"""
def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

"""
任务类型	    推荐方法	    适用场景
文本分类	    Spacy	        最精准
情感分析	    TextBlob	    简单易用
机器翻译	    Stanza	        支持多语言
搜索引擎优化     WordNet	    基础 NLP 任务
社交媒体 NLP	Pattern	        适用于短文本
"""


"""
normalize(words)
作用：依次调用多个预处理步骤对文本进行清理。
包含步骤：
remove_non_ascii（去除非 ASCII 字符）
to_lowercase（转换为小写）
remove_punctuation（去除标点）
replace_numbers（数字转英文）
remove_stopwords（去除停用词）
"""
def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

"""
stem_and_lemmatize(words)
作用：对单词执行词干提取（Stemming）和词形还原（Lemmatization）。
返回值：返回两个列表：
词干化后的单词列表
词形还原后的单词列表
"""
def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas


"""
完整的文本处理流程如下：

1. 文本清理
    denoise_text(text) 处理 HTML 代码和方括号内容。
    replace_contractions(text) 处理缩写词。
2. 单词级别处理
    normalize(words) 依次执行：
        去除非 ASCII 字符
        转换为小写
        去除标点符号
        数字转换为单词
        删除停用词
3. 文本标准化
    stem_and_lemmatize(words) 进行词干提取和词形还原。
"""

"""
文本分类	Tokenization, Lowercasing, Stopword Removal, Lemmatization
机器翻译	Sentence Splitting, Tokenization
情感分析	Tokenization, Stemming, Stopword Removal
语音识别	Lowercasing, Contractions Fixing
深度学习	Word Embeddings (Word2Vec, BERT)
"""