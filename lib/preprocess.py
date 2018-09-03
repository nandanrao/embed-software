import re
from sklearn.feature_extraction.text import strip_accents_ascii, strip_tags

# PATTERNS
############################
headlines = re.compile(r"#+\s*[^\n]+\n")
md_links = re.compile('\[[^\]]+\]\([^\)]+\)')
sentance = re.compile(r"\.\s+")
links = re.compile(r"https?://[^\s]+")
code_ticks = re.compile(r"``?`?[^`]+``?`?")
token_pattern = re.compile(r"(?u)\b\w\w+\b")
underscore = re.compile(r"\w*_\w*")
space = re.compile(r'\s+')
num = re.compile(r'[0-9]+')

tokenizer = lambda doc: token_pattern.findall(doc)

def claims_processor(s):
    # Lowercase
    s = s.lower()
    
    # Get rid of numbers in patents
    s = re.sub(num, '', s) if s else None
    
    # URLs and ASCII only
    s = re.sub(links, '', s)
    s = strip_accents_ascii(s)
    s = strip_tags(s)
    
    return s

def readme_processor(s):
    # Capitalization won't help us
    s = s.lower()

    # Remove code and markdown headlines
    s = re.sub(code_ticks, '', s)    
    s = re.sub(headlines, '', s)
    s = re.sub(md_links, '', s)
    s = re.sub(links, '', s)

    # ASCII our text and remove html tags
    s = strip_accents_ascii(s)
    s = strip_tags(s)

    # Underscores imply variable names, which are
    # never useful. Get rid of anything in camelcase? 
    s = re.sub(underscore, '', s)

    

class Preprocessor():
    def __init__(self, string_processor, min_words_per_sentence):
        self.string_processor = string_processor
        self.min_words_per_sentence = min_words_per_sentence
    
    def process(self, s):
        MIN_SENTENCES_PER_DOC = 2
        MIN_CHARS_PER_DOC = 25
        char_count = len(s)

        s = self.string_processor(s)

        # TODO: do make your "sentance" the whole paragraph? Closer to full document representation? 
        # Split on sentances, tokenize within the sentance, then replace 
        # sentance with \t separator for starspace/fasttext
        s = [i for i in sentance.split(s)]
        s = [tokenizer(i) for i in s]
        s = [' '.join(li_tokens) for li_tokens in s 
             if len(li_tokens) >= self.min_words_per_sentence]
        sentences = [i for i in s if i]

        s = '\t'.join(sentences)

        # Get rid of useless little documents
        if len(sentences) < MIN_SENTENCES_PER_DOC or char_count < MIN_CHARS_PER_DOC:
            return None

        return s