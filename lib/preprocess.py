import re
from sklearn.feature_extraction.text import strip_accents_ascii, strip_tags
import pandas as pd
from tqdm import tnrange, tqdm_notebook
from multiprocessing import Pool
import dataset
import gcsfs
import s3fs
from os.path import join

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

def claims_processor(s, numbers = False):
    # Lowercase
    s = s.lower()
    
    # Get rid of numbers in patents
    if numbers is False:
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
    return s
    

class Preprocessor():
    # For use in notebooks, provides tqdm notebook bar
    def __init__(self, 
                 string_processor, 
                 min_words_per_sentence, 
                 min_sentences_per_doc = 2,
                 min_chars_per_doc = 25,
                 **kwargs):
        self.string_processor = string_processor
        self.min_words_per_sentence = min_words_per_sentence
        self.min_sentences_per_doc = min_sentences_per_doc
        self.min_chars_per_doc = min_chars_per_doc
        self.string_processor_kwargs = kwargs
    
    def process(self, s):
        char_count = len(s)

        s = self.string_processor(s, **self.string_processor_kwargs)

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
        if (len(sentences) < self.min_sentences_per_doc or 
            char_count < self.min_chars_per_doc):
            
            return None

        return s


class ParallelProcessor():    
    jl_pattern = re.compile('.+\.jl$')
    csv_pattern = re.compile('.+\.csv$')

    def __init__(self, 
                 string_processor, 
                 inpath, 
                 outpath, 
                 content_key,                 
                 id_key,
                 pandas_kwargs = {},
                 cores = None,
                 fs = gcsfs.GCSFileSystem(project='open-source-software')):
        self.string_processor = string_processor
        self.inpath = inpath
        self.outpath = outpath
        self.id_key = id_key
        self.content_key = content_key
        self.pandas_kwargs = pandas_kwargs
        self.cores = cores
        self.fs = fs
        
    def _read_df(self, filename, file):
        if re.match(self.csv_pattern, filename):
            return pd.read_csv(file, **self.pandas_kwargs)
        elif re.match(self.jl_pattern, filename):
            return pd.read_json(file, lines=True, **self.pandas_kwargs)
        else:
             raise TypeError('Cannot parse file: {}'.format(filename))

    def _read(self, filename):
        with self.fs.open(join(self.inpath, filename)) as fi:            
            return self._read_df(filename, fi)
            
    def _get_files(self):
        files = self.fs.ls(self.inpath)
        files = [f.split('/')[-1] for f in files]
        return [f for f in files if f]
    
    def process(self,  
                       f,  
                       compression = 'gzip'):
        
        key = self.content_key
        df = self._read(f)    
        df = df[df[key].notna()].reset_index(drop=True)
        
        processed = df[key].map(self.string_processor)
        
        df['content'] = processed
        df = (df[(df.content.notna()) & (df[self.id_key].notna())]
              .reset_index(drop=True)
              .drop(key, 1))
    
        return df
        
    def process_all(self):
        files = self._get_files()
        pool = Pool(self.cores)
        conn = dataset.connect('sqlite:///{}'.format(self.outpath))
        table = conn['processed']        
        for df in tqdm_notebook(pool.imap(self.process, files), total=len(files)):
            for i,c in zip(df[self.id_key], df.content):
                table.insert_ignore({self.id_key: i, 'content': c}, [self.id_key])           
        pool.close()
        pool.join()
