import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
def usingLibraries():
    # 1) Make sure you have the punkt, stopwords, and wordnet corpora
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # 2) Your raw corpus
    corpus = """The cat sat on the mat.
    The dog barked at the cat.
    The cat and dog played in the garden.
    A quick brown fox jumps over the lazy dog.
    Dogs are loyal animals, and cats are independent."""

    # 3) Preprocess: tokenize into sentences, clean, lemmatize, remove stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    processed_sents = []
    for sent in sent_tokenize(corpus):
        # keep only letters, lowercase, split
        words = re.sub(r'[^a-zA-Z]', ' ', sent).lower().split()
        # lemmatize & remove stopwords
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        processed_sents.append(" ".join(words))

    # 4) Build count matrix
    cv = CountVectorizer()
    X = cv.fit_transform(processed_sents).toarray()       # shape: (n_sents, n_features)
    features = cv.get_feature_names_out()   
    print(features)               # list of feature (word) names
    print(X)
    # 5) Sum up counts for each feature across all sentences
    word_counts = X.sum(axis=0)                            # shape: (n_features,)
    
    # 6) Plot
    plt.figure(figsize=(10,6))
    plt.bar(features, word_counts)
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.title("Word Counts in Corpus")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
def fromScratch():
    corpus = """The cat sat on the mat.
    The dog barked at the cat.
    The cat and dog played in the garden.
    A quick brown fox jumps over the lazy dog.
    Dogs are loyal animals, and cats are independent."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    sent = corpus.split("\n")
    stop_words = set(stopwords.words('english'))
    all_words = []
    processed_sents = []
    for sent in sent_tokenize(corpus):
        # keep only letters, lowercase, split
        words = re.sub(r'[^a-zA-Z]', ' ', sent).lower().split()
        # lemmatize & remove stopwords
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        for word in words:
            if word not in all_words:
                all_words.append(word)
                print(word)
        processed_sents.append(" ".join(words))
    print(all_words)
    print(processed_sents)
    vector = []

    for sent in processed_sents:
        vec = [0 for x in range(len(all_words))]
        for word in nltk.word_tokenize(sent ): 
            for x in range(len(all_words)):    
                if word == all_words[x]:
                    print(word)
                    vec[x]=1
        vector.append(vec)

    print(vector)

    
fromScratch()
usingLibraries()