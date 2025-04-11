import spacy
#  Implementing different type of tokenization over several types of data

import nltk
data = """

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
nlp.add_pipe("sentencizer")

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




text='''
Look for data to help you address the question. Governments are good
sources because data from public research is often freely available. Good
places to start include http://www.data.gov/, and http://www.science.gov/, and in the United Kingdom, http://data.gov.uk/.
Two of my favorite data sets are the General Social Survey at http://www3.norc.org/gss+website/, 
and the European Social Survey at http://www.europeansocialsurvey.org/ .
'''

# TODO: Write code here
# Hint: token has an attribute that can be used to detect a url

transactions = "Tony gave two $ to Peter, Bruce gave 500 € to Steve"

transactions_nlp = nlp(transactions)
url_nlp = nlp(text)
url_list = [x for x in url_nlp]
data_list = [x for x in transactions_nlp]

for m in range(len(url_list)):
    if not url_list[m].like_url:
        continue
    print(url_list[m])
for m in range(len(data_list)):
    if not data_list[m].like_num and m == len(data_list):
        continue
    
    if data_list[m].like_num and data_list[m+1].is_currency:
        print(data_list[m],data_list[m+1])














