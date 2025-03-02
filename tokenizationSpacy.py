import spacy
#  Implementing different type of tokenization over several types of data

import nltk
data = """
Dr. Pappu
This is an example sentence.
Tokenization breaks text into smaller parts.
How many words are in this sentence?
Tokenizing text helps in natural language processing tasks.
ChatGPT can generate various forms of content.
Data preprocessing is an important step in NLP.
Machine learning models work with tokenized text.
Can you tokenize this sentence as well?
The weather is nice today, isn't it?
What are the benefits of tokenization in NLP?"""



word = nltk.word_tokenize(data)
sent = nltk.sent_tokenize(data)

print("                    Words                       ")
print("================================================")
print(word)
print("================================================")

print("                  Sentence                      ")
print("================================================")
print(sent)
print("================================================")
nlp = spacy.blank("en")

word_spacy = [token.text for token in  nlp(data)]
sent_spacy = [token.text for token in  nlp(data).sents]

print("                    Words                       ")
print("================================================")
print(word_spacy)
print("================================================")

print("                  Sentence                      ")
print("================================================")
print(sent_spacy)
print("================================================")











