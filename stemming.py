import requests
import nltk
import os
from nltk.stem import *
from nltk.corpus import *
from tabulate import tabulate
import pandas
import os

data = """
In the context of text preprocessing for natural language processing (NLP), techniques such as stemming, 
lemmatization, stop word removal, tokenization, and lowercasing are commonly applied to prepare raw text for analysis. 
Stop word removal eliminates common words (e.g., "the," "is," "at") that do not contribute significant meaning to the analysis, 
reducing the noise in the dataset. Tokenization breaks the text into individual words or phrases, enabling the model to process 
smaller units of meaning. Lowercasing standardizes the text by converting all characters to lowercase, ensuring consistency 
(e.g., "Apple" and "apple" are treated as the same word). Stemming reduces words to their root form by stripping prefixes and 
suffixes, while lemmatization improves on this by using dictionary-based rules to return words to their base form, considering 
context and grammatical role. Together, these techniques improve the efficiency and accuracy of text-based models by focusing 
on the most meaningful and relevant components of the data.
"""
class krovetzStemmer:

    def stem(word):
        try:
            return requests.get(os.getenv("API_URL")+"stem/"+word).json()["stemmed_word"]
        except:
            return word

def stemming(stemmer,sentences):
    header_row = ["sentence","stop words"," stem words"]
    org_vs_stem = ["Original Sentence","Stemmed Sentence"]
    org_vs_stem_data =[]
    data_rows = []
    data_html = "org                            ->                        stemmed"
    for sentence in sentences:
        new =""
        stop_words = []
        stem_words = []
        
        for word in nltk.word_tokenize(sentence):
            if word in stopwords.words("english"):
                stop_words.append(word)
            else:
                stem_words.append(stemmer.stem(word))
                new+=stemmer.stem(word)+" "
        data_html+="<br>"+sentence+"====>"+new
        data_rows.append([sentence,stop_words,stem_words])    
        org_vs_stem_data.append([sentence,new])
    html_content = f"""
    <html>
    <head>
        <title>Stemmed Sentences</title>
    </head>
    <body>
        <h1>Original vs Stemmed Sentences</h1>
        {data_html}
        <h1>Detailed Analysis</h1>
        {data_rows}
    </body>
    </html>
    """
    
    # Write HTML content to file
    with open("index.html", 'w') as file:
        file.write(html_content)
    os.startfile("index.html")

sentences = nltk.sent_tokenize(data)

stem = PorterStemmer()

snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()
krovetz = krovetzStemmer
stemming(stem,sentences)
input()
stemming(snowball,sentences)
input()
stemming(lancaster,sentences)
input()
stemming(krovetz,sentences)
input()
os.remove("index.html")





