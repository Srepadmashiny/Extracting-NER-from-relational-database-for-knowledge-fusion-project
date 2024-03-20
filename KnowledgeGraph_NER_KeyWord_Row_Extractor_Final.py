#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Name                       :   KnowledgeGraph_NER_KeyWord_Row_Extractor_Final.py
# Author                     :   K. Padmashiny
# Algo                       :   NER & Keyword Extraction
# Date                       :   29-May-2023
# Purpose                    :   NER & Keyword Extraction for GraphDB Mining                             
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# ### *Load Library & Init*
import time
import re
import pandas as pd
 
# text processing libraries
import re
import string
import nltk
from nltk.corpus import stopwords
 
stop_words = stopwords.words('english')
print(len(stop_words))
stop_words.extend(['bios',
'power',
'test',
'system',
'issue',
'charge',
'custom',
'value',
'button',
'adapter',
'behavior',
'platform',
'custom charge',
'code',
'recovery',
'solutions',
'function',
'case',
'solution',
'update',
'message',
'result',
'test case',
'process',
'list',
'event',
'time',
'type',
'pims',
'change',
'status',
'table',
'description',
'feature',
'option',
'note',
'p',
'error',
'test option',
'plan',
'job',
'end',
'tool',
'step',
'platforms',
'power',
'battery',
'feature',
'test',
'system',
'bios',
'supply',
'test cases',
'desktop',
'cases',
'enclosure',
'example',
'graphics',
'charge',
'os',
'test planning',
'planning',
'purposes',
'addition',
'phases',
'tablets',
'type',
'tcs',
'desktops',
'nbsp',
'states',
'processor',
'c',
'functionality',
'objective',
'station',
'non',
'solution',
'test case',
'time',
'management',
'systems',
'case',
'manager',
'people',
'options',
'status',
'behavior',
'end',
'point',
'update',
'period',
'option',
'error',
'results',
'issues',
'delivery',
'custom',
'ability',
'platforms',
'feature',
'bios',
'test',
'desktop',
'enclosure',
'cases',
'test cases',
'platform',
'audio',
'tcs',
'os',
'desktops',
'addition',
'test planning',
'purposes',
'example',
'planning',
'phases',
'setup',
'system',
'power',
'function',
'options',
'time',
'loss',
'error',
'option',
'data',
'check',
'errors',
'events',
'code',
'battery',
'user',
'values',
'field'
])

print(len(stop_words))
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import textacy

stop_words
# ### _NER & KeyWord Extraction_
import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
rules = {
    "Noun and compound": [
        {
            "DEP": "compound",
            "OP": "?"
        },
        {
            "POS": "NOUN"
        }
    ],
   
    "Noun": [
        {
            "POS": "NOUN"
        }
    ],
    "Noun and PRON": [
        {
            "POS": "NOUN"
        },
        {
            "POS": "PRON"
        }
    ],
    "Noun and adjective": [
        {
            "POS": "NOUN"
        },
        {
            "POS": "ADJ"
        }
    ]
}
rule_matcher = Matcher(nlp.vocab)
for rule_name, rule_tags in rules.items(): # register rules in matcher
    rule_matcher.add(rule_name, [rule_tags])
from rake_nltk import Rake #Rapid Keyword Extraction
rake_nltk_var = Rake()

# ### *Text PreProcess*
Short_Dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not",
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(Short_Dict.keys()))

# Function for Expanding Contractions
def Expand_Short_Dict(text,Short_Dict=Short_Dict):
  def replace(match):
    return Short_Dict[match.group(0)]
  return contractions_re.sub(replace, text)
 

def remove_non_english(a_str):
    ascii_chars = set(string.printable)
    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )
 

def clean_text(text):
    '''Make  lowercase, remove text in
            square brackets,
            remove links,
            remove punctuation, and
            remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    #text = re.sub('<.*?>+', ' ', text)
    text = re.sub(':', ' ', text)
    #text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    #text = re.sub('\w*\d\w*', ' ', text)
    return text
 
def text_preprocessing(text):
    """  Cleaning and parsing the text. """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stop_words]
    combined_text = ' '.join(tokenized_text)
    return combined_text
def extract(text):
    doc = nlp(text)  # Convert string to spacy 'doc' type
    matches = rule_matcher(doc)  # Run matcher
    result = []
    for match_id, start, end in matches:  # For each attribute detected, save it in a list
        attribute = doc[start:end]
        result.append(attribute.text)
    return result
 

 

# ### _File Input_
# --- Give the CSV fileName without extention

DirName = 'C:\\'

# --- Engineering Nudd

filename = "Engineering"
txt_cols = ['summary']

# ### *Data Loading & Text Pre Processing*
df = pd.DataFrame()
df = pd.read_csv(DirName + filename + '.csv', sep=',',low_memory=False, encoding = "ISO-8859-1"  , na_values=' ')
for key_col in txt_cols:
    df[key_col] = df[key_col].replace({'/':' '}, regex=True)
    df[key_col] = df[key_col].replace({'_':' '}, regex=True)
    df[key_col] = df[key_col].astype(str).str.replace("[@#/$]","" ,regex=True).astype(str)
    df[key_col] = df[key_col].replace({'Ã¢':''}, regex=True)
    print (key_col)
    #df[key_col] = df[key_col] .apply(lambda x:remove_non_english(x))
    #f[key_col] = df[key_col].str.replace(r'<[^<>]*>', ' ', regex=True)
    df[key_col] = df[key_col] .apply(lambda x:Expand_Short_Dict(x))
    df[key_col] = df[key_col].apply(str).apply(lambda x: text_preprocessing(x))
    if key_col  != 'summary' :
        df['summary'] = df['summary'] + ".  " + df[key_col]
 
df['summary'] = df['summary'].str.encode('ascii', 'ignore').str.decode('ascii')

 

df['KeyWords'] =""
for index, row in df.iterrows():
    rake_nltk_var.extract_keywords_from_text(df['summary'][index])
    #df['KeyWords'][index] = rake_nltk_var.get_ranked_phrases()
    tmp = extract(df['summary'][index])
    #df['KeyWords'][index] = df['KeyWords'][index] + tmp
    df['KeyWords'][index] = tmp


# ### _Save Results to a CSV_
df.columns = [x.encode('utf-8').decode('ascii', 'ignore') for x in df.columns]
print(df.columns)

df1 = df.explode('KeyWords')

#df1 = df1[~df1['KeyWords'].isin(stop_words)]  #Comented Smitha Change as per 24-Jul2023 Morning
df.to_csv(DirName + filename + '_KW.csv', index=False)
df1 = df1[['code','KeyWords']]
df1.to_csv(DirName + filename + '_KW_ROW.csv', index=False)
