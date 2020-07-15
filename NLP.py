#!/usr/bin/env python
# coding: utf-8

# Natural Langugage Processing 
# 
# Install spaCy first. To do so, go to https://spacy.io/usage. 
# Then, complete the quickstart questionnaire with the following answers according to your system and project's objectives. 
# 
# On my work laptop, I used windows and the anaconda prompt and then answered "y" after installing spaCy into my Anaconda Prompt (anaconda3) terminal. 
# 
# Many tutorials/online code/online guides do not go through the process above and so I am noting it there so that you can successfully rerun this notebook. 

# !pip install -U spacy download en_core_web_sm
# https://anaconda.org/conda-forge/spacy-model-en_core_web_sm
# 

# In[3]:


import spacy
nlp = spacy.load("en_core_web_sm")


# In[5]:


doc = nlp("Coffee wakes me up the most, what do you think?")
for token in doc:
    print(token)


# In[8]:


print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")


# In[9]:


from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')


# In[16]:


terms = ['Apple', 'Facebook', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", None, *patterns)


# In[17]:


# Sample review
text_doc = nlp("Glowing review overall, and some really interesting side-by-side via Apple and Facebook "
               "photography tests pitting the iPhone XS against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 
matches = matcher(text_doc)
print(matches)


# In[18]:


match_id, start, end = matches[0]
print(nlp.vocab.strings[match_id], text_doc[start:end])

