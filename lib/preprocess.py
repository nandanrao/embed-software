import re
from sklearn.feature_extraction.text import strip_accents_ascii, strip_tags

headlines = re.compile(r"#+\s*[^\n]+\n")
md_links = re.compile('\[[^\]]+\]\([^\)]+\)')
sentance = re.compile(r"\.\s+")
links = re.compile(r"https?://[^\s]+")
code_ticks = re.compile(r"``?`?[^`]+``?`?")
token_pattern = re.compile(r"(?u)\b\w\w+\b")
tokenizer = lambda doc: token_pattern.findall(doc)
underscore = re.compile(r"\w*_\w*")
space = re.compile(r'\s+')

def preprocessor(s):

    char_count = len(s)
    
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


    # Split on sentances, tokenize within the sentance, then replace 
    # sentance with \t separator for starspace/fasttext
    s = [i for i in sentance.split(s)]
    s = [' '.join(tokenizer(i)) for i in s]
    sentences = [i for i in s if i]
    s = '\t'.join(sentences)

    # Get rid of useless little readmes
    if len(sentences) < 2 or char_count < 25:
        return None

    return s