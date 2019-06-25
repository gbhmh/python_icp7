import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from  nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk import wordpunct_tokenize,pos_tag,ne_chunk


file_content = open("input.txt").read()
wtokens = nltk.word_tokenize(file_content)

for t in wtokens:
  print(t)

print(nltk.pos_tag(wtokens))

pStemmer = PorterStemmer()
for t in wtokens:
  print(t, pStemmer.stem(t.lower()))

lStemmer = LancasterStemmer()
for t in wtokens:
  print(t, lStemmer.stem(t.lower()))
  
snowball = SnowballStemmer('english')
for t in wtokens:
  print(t, snowball.stem(t.lower()))

lemmatizer = WordNetLemmatizer()
for t in wtokens:
  print(t, lemmatizer.lemmatize(t.lower()))

trigrams = ngrams(wtokens,3)
print(Counter(trigrams))

# named entity recognition
print(ne_chunk(pos_tag(wordpunct_tokenize(file_content))))
