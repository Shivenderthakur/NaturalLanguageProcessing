import regex
import nltk
import spacy
nltk.download("punkt")
nltk.download("treebank")

class RegEx:
    def __init__(self):
        self.patterns={
    "NOUN": [
        r"\b[A-Z][a-z]*\b",  # Proper nouns (start with capital)
        r"\b[a-z]+(?:'[a-z]+)?\b",  # Common nouns (including possessives)
        r"\b[a-z]+(?:-[a-z]+)+\b",  # Compound nouns (hyphenated)
        r"\b[a-z]+s\b",  # Plural nouns (simple -s)
        r"\b[a-z]+es\b",  # Plural nouns (simple -es)
        r"\b[a-z]+ies\b",  # Plural nouns (-y to -ies)
        r"\b[A-Z]+\b",  # Acronyms
        r"\b\d+(?:st|nd|rd|th)?\s+(?:century|decade|year)\b",  # Time-based nouns
        r"\b[a-z]+(?:ing|ment|tion|sion|ness|ity|ance|ence)\b"  # Nouns formed from other word types
    ],
    "PRONOUN": [
        r"\b(?:I|you|he|she|it|we|they|me|him|her|us|them|mine|yours|his|hers|its|ours|theirs|myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|who|whom|whose|which|that|someone|anyone|everyone|no one|something|anything|everything|nothing)\b",
        r"\b(?:this|that|these|those)\b"  # Demonstrative pronouns
    ],
    "VERB": [
        r"\b[a-z]+(?:ed|ing|s|es|ies)\b",  # Regular verb forms
        r"\b(?:am|is|are|was|were|be|being|been|have|has|had|do|does|did|can|could|may|might|must|shall|should|will|would)\b",  # Auxiliary verbs
        r"\b[a-z]+(?:ize|ise|ate|ify)\b",  # Verbs ending in common suffixes
        r"\b[a-z]+(?:en)\b",  # Verbs ending in 'en'
        r"\bto\s+[a-z]+\b"  # Infinitive form
    ],
    "ADJECTIVE": [
        r"\b[a-z]+(?:able|ible|al|ial|ful|less|ous|ive|ic|ish|y|ed|ing)\b",  # Common adjective suffixes
        r"\b[a-z]+(?:er|est)\b",  # Comparative and superlative forms
        r"\b(?:good|better|best|bad|worse|worst|little|less|least|many|more|most|few|fewer|fewest)\b",  # Irregular adjectives
        r"\b(?:large|small|big|tiny|huge|enormous|red|blue|green|yellow|black|white|old|new|young|happy|sad|angry|calm)\b",  # Common adjectives
        r"\b[A-Z][a-z]+(?:an|ian|ese|ish)\b"  # Adjectives derived from places
    ],
    "ADVERB": [
        r"\b[a-z]+ly\b",  # Most common adverb form
        r"\b(?:very|quite|really|too|so|just|almost|always|never|often|sometimes|now|then|here|there|well|fast|slowly)\b",  # Common adverbs
        r"\b(?:more|less|most|least)\s+[a-z]+\b"  # Comparative/superlative adverbs with more/less/most/least
    ],
    "PREPOSITION": [
        r"\b(?:in|on|at|by|to|from|with|about|above|across|after|against|along|among|around|as|before|behind|below|beneath|beside|between|beyond|during|except|for|inside|into|near|of|off|onto|outside|over|since|through|under|until|up|via)\b"
    ],
    "CONJUNCTION": [
        r"\b(?:and|or|but|because|although|though|if|unless|while|since|for|so|yet|nor|either|neither|whether|that)\b"
    ],
    "INTERJECTION": [
        r"\b(?:oh|ah|wow|oops|hey|alas|hurray|yeah|phew|ugh|ouch)\b",
        r"\b[A-Z]+\b(?:!|\?)\b"  # Interjections that are capitalized and end with ! or ?
    ],
    "DETERMINER": [
        r"\b(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their|all|any|both|each|either|every|few|many|more|most|no|some|such|what|which|whose)\b",
        r"\b\d+\b"  # Numbers as determiners
    ] ,
    "NUMERAL": [
        r"\b\d+(?:\.\d+)?\b",  # Integers and decimals
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion)\b",  # Word form numerals
        r"\b\d+(?:st|nd|rd|th)\b",  # Ordinal numbers
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"  # Word form ordinal numbers
    ],
    "PARTICLE": [
        r"\b(?:to|up|down|in|out|off|on|away|back|over|under|again|further|just|only|still|yet|even|ever|never|always|already|so|too|as|about|around|before|after|besides|between|by|concerning|during|except|for|from|inside|into|near|of|out|outside|over|through|till|toward|under|until|upon|within|without)\b",
        r"\b(?:n't|n't\')\b"  # Contraction of 'not' (e.g., isn't, don't)
    ],
    "UNKNOWN": [
        r"\b(?:[A-Za-z]+|[0-9]+)\b"  # Catch-all for unknown words
    ],
    "SYM": [
        r"\b(?:[\+\-\*/=<>!&|^%$#@~`_]+|(?:\d+[\.,]?\d*)?[\.,;:!?])\b"
    ]
    }
class NltkTaggersRuleBased(RegEx):
    def __init__(self,doc,sentence=False):
        """
        if the document is 
        ->txt file then read in raw form and put in doc
        ->sentence just put it same with sentecne=True
        ->para just put is
        """
        self.sentences =[]
        self.train = nltk.corpus.treebank.tagged_sents()[:]
        self.words=[]
        if not sentence:
            for x in nltk.tokenize.sent_tokenize(doc):
                self.sentences.append(x)
                for y in nltk.tokenize.word_tokenize(x):
                    self.words.append(y)
        else:
            self.sentences.append(sentence)
            for y in nltk.tokenize.word_tokenize(sentence):
                self.words.append(y) 
        self.tagger_tagged = dict()
        self.lookuptaggers = [nltk.tag.BigramTagger,nltk.tag.UnigramTagger,nltk.tag.TrigramTagger]
        self.transformationtagger = [nltk.tag.BrillTagger]
    def startTagging(self):
        
        for tagger in self.lookuptaggers:
            
            tagger_w=tagger(self.train)
            self.tagger_tagged[tagger]=[tagger_w.accuracy(self.sentences,backoff=nltk.tag.DefaultTagger("NN"))]
            
                    
sms = NltkTaggersRuleBased(doc="""In the context of text preprocessing for natural language processing (NLP), techniques such as stemming, 
lemmatization, stop word removal, tokenization, and lowercasing are commonly applied to prepare raw text for analysis. 
Stop word removal eliminates common words (e.g., "the," "is," "at") that do not contribute significant meaning to the analysis, 
reducing the noise in the dataset. Tokenization breaks the text into individual words or phrases, enabling the model to process 
smaller units of meaning. Lowercasing standardizes the text by converting all characters to lowercase, ensuring consistency 
(e.g., "Apple" and "apple" are treated as the same word). Stemming reduces words to their root form by stripping prefixes and 
suffixes, while lemmatization improves on this by using dictionary-based rules to return words to their base form, considering 
context and grammatical role. Together, these techniques improve the efficiency and accuracy of text-based models by focusing 
on the most meaningful and relevant components of the data.""")          
sms.startTagging()



