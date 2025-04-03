import re,nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import matplotlib.pyplot as plt


# Calculate the sum of each feature (word) across all sentences
word_counts = x.sum(axis=0)


# Create the bar graph

corpus =  """The cat sat on the mat.
    The dog barked at the cat.
    The cat and dog played in the garden.
    A quick brown fox jumps over the lazy dog.
    Dogs are loyal animals, and cats are independent. """
corpusd = []
sent = sent_tokenize(corpus)
for i in range(len(sent)):
    val = re.sub("[^a-zA-Z]",' ',sent[i])
    val = val.lower()
    val = val.split()

    val = [WordNetLemmatizer().lemmatize(word) for word in val if val not in stopwords.words("english")]
    
    val = " ".join(val)
    print(val)
    corpusd.append(val)
    


cv = CountVectorizer()
x = cv.fit_transform(corpusd).toarray()
f = cv.get_feature_names_out()
y = np.transpose(x)
plt.figure(figsize=(10, 6))
plt.bar(f, word_counts)
plt.xlabel("Words")
plt.ylabel("Count")
plt.title("Word Counts in Corpus")
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()