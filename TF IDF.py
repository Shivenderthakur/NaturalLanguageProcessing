import nltk
import time
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
import re

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TF_IDF:

    def __init__(self, document):
        print("Processing started at:", time.ctime())

        self.doc = document
        self.sents = nltk.sent_tokenize(self.doc)
        self.processed_sents = []
        self.Vocublary = []

        lemmatizer = nltk.WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        for sent in self.sents:
            words = re.sub(r'[^a-zA-Z]', ' ', sent).lower().split()
            words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
            for word in words:
                if word not in self.Vocublary:
                    self.Vocublary.append(word)
            self.processed_sents.append(" ".join(words))

        self.vocabTFDictionary = pd.DataFrame(columns=["Words"] + self.processed_sents)
        self.vocabIDFDictionary = pd.DataFrame(columns=["Words", "IDF"])
        self.vocabTFIDFDictionary = pd.DataFrame(columns=["Words"] + self.processed_sents)

        self.countTermFrequency()
        self.countInverseDocumentFrequency()
        self.countTFIDF()

    def countTermFrequency(self):
        for word in self.Vocublary:
            row = {"Words": word}
            for sent in self.processed_sents:
                tokens = nltk.word_tokenize(sent)
                term_freq = tokens.count(word) / len(tokens) if len(tokens) > 0 else 0
                row[sent] = term_freq
            row_df = pd.DataFrame([row])
            self.vocabTFDictionary = pd.concat([self.vocabTFDictionary, row_df], ignore_index=True)

    def countInverseDocumentFrequency(self):
        total_sents = len(self.processed_sents)

        for word in self.Vocublary:
            count = sum(1 for sent in self.processed_sents if word in nltk.word_tokenize(sent))
            idf_value = np.log(total_sents / (count + 1))  # +1 to avoid division by zero
            row = {"Words": word, "IDF": idf_value}
            row_df = pd.DataFrame([row])
            self.vocabIDFDictionary = pd.concat([self.vocabIDFDictionary, row_df], ignore_index=True)

    def countTFIDF(self):
        for _, tf_row in self.vocabTFDictionary.iterrows():
            word = tf_row["Words"]
            idf_row = self.vocabIDFDictionary[self.vocabIDFDictionary["Words"] == word]
            idf_value = float(idf_row["IDF"].values[0]) if not idf_row.empty else 0.0

            tfidf_row = {"Words": word}
            for sent in self.processed_sents:
                tf = tf_row[sent]
                tfidf_row[sent] = tf * idf_value

            row_df = pd.DataFrame([tfidf_row])
            self.vocabTFIDFDictionary = pd.concat([self.vocabTFIDFDictionary, row_df], ignore_index=True)


vect = TF_IDF("""
The king and queen ruled with wisdom, ensuring peace and prosperity in their kingdom. 
The queen’s generosity and the king’s bravery in battle earned them the love of their people. 
Their children, the prince and princess, traveled far, learning from other rulers and forming alliances. 
They saw the rise of football and technology, both embraced by the kingdom for their unifying potential. 
The royal family valued education, art, and scientific progress, drawing inspiration from history and fostering growth. 
The king and queen took action against climate change, ensuring a sustainable future. 
They also navigated the challenges of social media, promoting positive connections. 
Their reign was a blend of tradition and innovation, ensuring the kingdom thrived.
""")

print("\nTerm Frequency (TF):")
print(vect.vocabTFDictionary)

print("\nInverse Document Frequency (IDF):")
print(vect.vocabIDFDictionary)

print("\nTF-IDF:")
print(vect.vocabTFIDFDictionary)


