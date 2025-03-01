import spacy

sentence = "The quick brown fox jumps over 12 lazy dogs! @#&*()_+{}[]|:;,.<>?/~`"
paragraph = """
Tokenization is a critical process in natural language processing (NLP) that involves breaking down a stream of text into smaller, discrete units known as tokens. These tokens can be words, punctuation marks, special characters, numbers, or even spaces, depending on the rules and context. A key aspect of tokenization is that it must handle every printable character in the input, ensuring that each token is valid and meaningful. This includes dealing with special characters like punctuation marks (e.g., periods, commas, exclamation points), symbols (e.g., @, #, $, %, &), mathematical operators, and even white spaces. For example, a tokenizer might separate a sentence like "I have 5 apples!" into tokens like "I", "have", "5", "apples", and "!". 

Special characters often require careful handling to preserve the integrity of the text. Some tokenizers treat punctuation as individual tokens, while others may attach it to adjacent words depending on the use case, such as handling contractions (e.g., "don't" as "don" and "t"). Non-alphanumeric symbols, like hashtags (#) or at signs (@), may be tokenized as part of the word in contexts like social media analysis. Additionally, the tokenization process must respect non-standard characters, including emojis, currency symbols, and diacritical marks used in various languages, which may also be treated as separate tokens. Handling every printable and valid character in a consistent way ensures the text is properly processed for further NLP tasks like sentiment analysis, machine translation, and text classification. In essence, robust tokenization accounts for the diverse range of characters in the text and prepares the data for downstream tasks while maintaining linguistic accuracy and integrity.
"""

nlp = spacy.blank("en")
doc_sen = nlp(sentence)
doc_para = nlp(paragraph)
for y in [doc_sen,doc_para]:
    print("=====================================================================\n\n\n")
    for x in y:
        print(x)
    print("=====================================================================\n\n\n")
for m in [nlp,doc_sen,x]:
    print(type(m))

    