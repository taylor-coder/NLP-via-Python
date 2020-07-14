#!/usr/bin/env python
# coding: utf-8

# Antibiotics Report (Work in progress)
# 
# Natural Language Processing / Text Analysis Project
# 
# This jupyter notebook / github project repository details the process of text analytics. In particular, it shows the process of extracting and cleaning text from a PDF. It also shows how to pass text through an NLP pipeline while using graphs to present the results.
# 
# Install: 
# conda install -c conda-forge pypdf2
# conda install -c conda-forge spacy
# conda install -c anaconda numpy
# conda install -c anaconda pandas
# conda install -c conda-forge matplotlib
# conda install seaborn
# conda install -c conda-forge geopandas
# conda install -c conda-forge descartes

# In[1]:


import PyPDF2
import spacy
import numpy
import pandas
import en_core_web_sm


# In[2]:


reader = PyPDF2.PdfFileReader("2019-ar-threats-report-508.pdf")
full_text = ""


# In[3]:


pdf_page_number = 7


# In[4]:


full_text = full_text.replace("  ", " ").replace(
    "  ", " ").replace("  ", " ")

with open("transcript_clean.txt", "w", encoding="utf-8") as temp_file:
    temp_file.write(full_text)


# In[5]:


corpus = open("transcript_clean.txt", "r", encoding="utf-8").read()
nlp = spacy.load("en_core_web_sm")


# In[6]:


nlp.max_length = len(corpus)
doc = nlp(corpus)


# In[7]:


data_list = [["text", "text_lower", "lemma", "lemma_lower",
                "part_of_speech", "is_alphabet", "is_stopword"]]

for token in doc:
    data_list.append([token.text, token.lower_, token.lemma_, token.lemma_.lower(), token.pos_, token.is_alpha, token.is_stop])


# In[8]:


import csv
from textblob_de import TextBlobDE as TextBlob


# In[9]:


csv.writer(open("./tokens.csv", "w", encoding="utf-8",
                newline="")).writerows(data_list)


# In[10]:


data_list = [["text", "text_lower", "label"]]

for ent in doc.ents:
    data_list.append([ent.text, ent.lower_, ent.label_])

csv.writer(open("./entities.csv", "w", encoding="utf-8",
                newline="")).writerows(data_list)


# Negative Words in English TXT Source
# https://gist.github.com/mkulakowski2/4289441

# In[11]:


with open("positivewords.txt", "r", encoding="utf-8") as temp_file:
    positivewords = temp_file.read().splitlines()

with open("negativewords.txt", "r", encoding="utf-8") as temp_file:
    negativewords = temp_file.read().splitlines()


# In[12]:


data_list = [["text", "score"]]

for sent in doc.sents:

    # Only take into account real sentences.
    if len(sent.text) > 10:

        score = 0

        # Start scoring the sentence.
        for word in sent:

            if word.lower_ in positive_words:
                score += 1

            if word.lower_ in negative_words:
                score -= 1

        data_list.append([sent.text, score])

csv.writer(open("./sentences.csv", "w", encoding="utf-8",
                newline="")).writerows(data_list)


# In[13]:


import seaborn as sns
import numpy as np 
import pandas as pd


# In[14]:


sns.set(style="ticks",
    rc={
        "figure.figsize": [12, 7],
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.facecolor": "#5C0E10",
        "figure.facecolor": "#5C0E10"}
    )


# In[15]:



"""
Visit this website for additional details regarding Spacy: https://spacy.io/usage/spacy-101

This script extracts features from the transcript txt file and saves them to .csv files
so they can be used in any toolkkit.
"""

import csv
import spacy


def main():
  """Loads the model and processes it.
  
  The model used can be installed by running this command on your CMD/Terminal:
  python -m spacy download es_core_news_md
  
  """

  corpus = open("transcript_clean.txt", "r", encoding="utf-8").read()
  nlp = spacy.load("en_core_web_sm")

  # Our corpus is bigger than the default limit, we will set
  # a new limit equal to its length.
  nlp.max_length = len(corpus)

  doc = nlp(corpus)

  get_tokens(doc)
  get_entities(doc)
  get_sentences(doc)


def get_tokens(doc):
  """Get the tokens and save them to .csv
  Parameters
  ----------
  doc : spacy.doc
      A doc object.
  """

  data_list = [["text", "text_lower", "lemma", "lemma_lower",
                "part_of_speech", "is_alphabet", "is_stopword"]]

  for token in doc:
      data_list.append([
          token.text, token.lower_, token.lemma_, token.lemma_.lower(),
          token.pos_, token.is_alpha, token.is_stop
      ])

  with open("./tokens.csv", "w", encoding="utf-8", newline="") as tokens_file:
      csv.writer(tokens_file).writerows(data_list)


def get_entities(doc):
  """After getting the entitites they are saved to a .csv 
  using the codes below...
  
  Code Parameters
  ----------
  doc : spacy.doc 
      A doc object.
  """

  data_list = [["text", "text_lower", "label"]]

  for ent in doc.ents:
      data_list.append([ent.text, ent.lower_, ent.label_])

  with open("./entities.csv", "w", encoding="utf-8", newline="") as entities_file:
      csv.writer(entities_file).writerows(data_list)


def get_sentences(doc):
  """Get the sentences, score and save them to .csv
  You will require to download the dataset (zip) from the following url:
  
  You will need to download the txt for both positive and negative 
  words in English, save to your computer, and upload to Jupyter Notebook
  or whatever interface you are coding in Python with. 
  negativewords.txt
  positivewords.txt
  Parameters
  ----------
  doc : spacy.doc
      A doc object.
  """

  # Load positive and negative words into lists.
  with open("positivewords.txt", "r", encoding="utf-8") as temp_file:
      positivewords = temp_file.read().splitlines()

  with open("negativewords.txt", "r", encoding="utf-8") as temp_file:
      negativewords = temp_file.read().splitlines()

  data_list = [["text", "score"]]

  for sent in doc.sents:

      # Only take into account real sentences.
      if len(sent.text) > 10:

          score = 0

          # Start scoring the sentence.
          for word in sent:

              if word.lower_ in positive_words:
                  score += 1

              if word.lower_ in negative_words:
                  score -= 1

          data_list.append([sent.text, score])


  with open("./sentences.csv", "w", encoding="utf-8", newline="") as sentences_file:
      csv.writer(sentences_file).writerows(data_list)

import sys

if __name__ == "__main__":

  main()


# In[16]:


df = pd.read_csv("./tokens.csv")


# In[17]:


df.loc[df["lemma_lower"] == "programa", "lemma_lower"] = "programar"


# In[18]:


words = df[(df["is_alphabet"] == True) & (df["is_stopword"] == False) & (
    df["lemma_lower"].str.len() > 1)]["lemma_lower"].value_counts()[:20]


# In[19]:


import PyPDF2
import re
pdfFileObj = open('2019-ar-threats-report-508.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfReader.numPages


# In[20]:


pageObj = pdfReader.getPage(0)
pageObj.extractText()


# In[21]:


object = PyPDF2.PdfFileReader("2019-ar-threats-report-508.pdf")


# In[22]:


# Page number code - Python 
NumPages = object.getNumPages()


# In[23]:


String = "antibiotic"


# In[24]:


for i in range(0, NumPages):
    PageObj = object.getPage(i)
    print("this is page " + str(i)) 
    Text = PageObj.extractText() 
    # print(Text)
    ResSearch = re.search(String, Text)
    print(ResSearch)


# In[25]:


import os 


# In[26]:


from os import path
from wordcloud import WordCloud


# In[27]:


d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()


# In[28]:


pdfReader = PyPDF2.PdfFileReader(open('2019-ar-threats-report-508.pdf', 'rb'))
pageData = ''
for page in pdfReader.pages:
    pageData += page.extractText()
    print(pageData)


# In[29]:


reader = PyPDF2.PdfFileReader("2019-ar-threats-report-508.pdf")
full_text = ""


# In[30]:


pdf_page_number = 7


# In[31]:


for i in range(1, 150):
    # This block is used to remove the page number at the start of
    # each page. The first if removes page numbers with one digit.
    # The second if removes page numbers with 2 digits and the else
    # statement removes page numbers with 3 digits.
    if pdf_page_number <= 9:
        page_text = reader.getPage(i).extractText().strip()[1:]
    elif pdf_page_number >= 10 and pdf_page_number <= 99:
        page_text = reader.getPage(i).extractText().strip()[2:]
    else:
        page_text = reader.getPage(i).extractText().strip()[3:]

    full_text += page_text.replace("\n", "")
    pdf_page_number += 1


# In[ ]:





# In[33]:


# This looks weird but that's the most practical way to remove double to quad white spaces.
full_text = full_text.replace("  ", " ").replace(
    "  ", " ").replace("  ", " ")

with open("transcript_clean.txt", "w", encoding="utf-8") as temp_file:
    temp_file.write(full_text)


# In[34]:


corpus = open("transcript_clean.txt", "r", encoding="utf-8").read()
nlp = spacy.load("en_core_web_sm")

# Our corpus is bigger than the default limit, we will set
# a new limit equal to its length.
nlp.max_length = len(corpus)

doc = nlp(corpus)


# In[35]:


data_list = [["text", "text_lower", "lemma", "lemma_lower",
                "part_of_speech", "is_alphabet", "is_stopword"]]

for token in doc:
    data_list.append([token.text, token.lower_, token.lemma_, token.lemma_.lower(), token.pos_, token.is_alpha, token.is_stop])

csv.writer(open("./tokens.csv", "w", encoding="utf-8",
                newline="")).writerows(data_list)


# In[36]:


data_list = [["text", "text_lower", "label"]]

for ent in doc.ents:
    data_list.append([ent.text, ent.lower_, ent.label_])

csv.writer(open("./entities.csv", "w", encoding="utf-8",
                newline="")).writerows(data_list)


# from wordcloud import WordCloud, STOPWORDS 
# import matplotlib.pyplot as plt 
# import pandas as pd 
# df = pd.read_csv(r"2019-ar-threats-report-508.csv", encoding ="latin-1") 
# comment_words = '' 
# stopwords = set(STOPWORDS) 
# import csv
# from wordcloud import WordCloud
# your_list = []
# with open('2019-ar-threats-report-508.csv', 'rb') as f:
#     reader = csv.reader(f)
#     your_list = '\t'.join([i[0] for i in reader])
# wordcloud = WordCloud().generate(your_list)

# In[37]:


reader = PyPDF2.PdfFileReader('2019-ar-threats-report-508.pdf')


# In[38]:


from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


# In[39]:


def convert_pdf_to_string(file_path):

	output_string = StringIO()
	with open(file_path, 'rb') as in_file:
	    parser = PDFParser(in_file)
	    doc = PDFDocument(parser)
	    rsrcmgr = PDFResourceManager()
	    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
	    interpreter = PDFPageInterpreter(rsrcmgr, device)
	    for page in PDFPage.create_pages(doc):
	        interpreter.process_page(page)

	return(output_string.getvalue())


# In[40]:


def convert_title_to_filename(title):
    filename = title.lower()
    filename = filename.replace(' ', '_')
    return filename


# In[41]:


def split_to_title_and_pagenum(table_of_contents_entry):
    title_and_pagenum = table_of_contents_entry.strip()
    
    title = None
    pagenum = None
    
    if len(title_and_pagenum) > 0:
        if title_and_pagenum[-1].isdigit():
            i = -2
            while title_and_pagenum[i].isdigit():
                i -= 1

            title = title_and_pagenum[:i].strip()
            pagenum = int(title_and_pagenum[i:].strip())
        
    return title, pagenum


# Below are more works in progress...

# writer = PyPDF2.PdfFileWriter()
# 
# for page in range(2,4):
# 
#     writer.addPage(reader.getPage(page))
#     
# output_filename = './data/original/table_of_contents.pdf'
# 
# with open(output_filename, 'wb') as output:
#     writer.write(output)
#     
# text = data_func.convert_pdf_to_string(
#     './data/original/table_of_contents.pdf')

# In[44]:


sns.set(style="ticks",
    rc={
        "figure.figsize": [12, 7],
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.facecolor": "#5C0E10",
        "figure.facecolor": "#5C0E10"}
    )


# In[45]:


df = pd.read_csv("2019-ar-threats-report-508.csv")


# In[46]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
df = pd.read_csv('2019-ar-threats-report-508.csv')
df.head()


# In[47]:


df.isnull().sum()


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re
import spacy
from spacy.lang.en import English
parser = English()


# In[49]:


STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
class CleanTextTransformer(TransformerMixin):
   def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
   def fit(self, X, y=None, **fit_params):
        return self
def get_params(self, deep=True):
        return {}
    
def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text
def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


# Please let me know if there are any questions and/or concerns. This is a work in progress project... 
