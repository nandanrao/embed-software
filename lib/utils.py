import requests
import re
import numpy as np
import linecache
from .preprocess import Preprocessor, readme_processor
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
import subprocess 
from copy import deepcopy
from hashlib import md5
from diskcache import FanoutCache
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

cache = FanoutCache('repo_cache/fanoutcache')
REPO_STOP_WORDS = ENGLISH_STOP_WORDS | frozenset(['et', 'al', 'pdf', 'star'])



def get_id(i):
    return linecache.getline('./prepared-readmes/file_lookup.csv', i+1).split(',')[0]

def get_ids(locs):
    return [get_id(i) for i in locs]
    
    
def call_ss(commands, input):
    p = subprocess.Popen(commands, 
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, encoding='utf-8')    

    o,e = p.communicate(input=input)
    return o,e    

def query_predict(model_path, k, input):
    out,err = call_ss(['query_predict'] + [model_path, str(k)], input)
    arr = out.split('\n')[28:28+k]
    arr = [a.split(':')[-1] for a in arr]
    arr = [re.search(r'\d+', a)[0] for a in arr]
    return [int(a) for a in arr]
    
def embed_docs(model_path, input):
    o,e = call_ss(["embed_doc", model_path], input)
    return np.array(get_ss_embed(o))


@cache.memoize(typed=True, expire=None, tag='readme')
def get_readme(repo, attempts = 0):
    """ Get readmes from Github and run through our preprocessor """
    names = ['README', 'readme', 'Readme']
    endings = ['.md', '', '.txt', '.markdown', '.rst']
    filenames = [n+e for n in names for e in endings]
    try: 
        f = filenames[attempts]
    except IndexError:
        print('REPO 404: {}'.format(repo))
        return None
    r = requests.get('https://raw.githubusercontent.com/{}/master/{}'.format(repo, f))
    if r.status_code == 404:
        return get_readme(repo, attempts + 1)
    preprocessor = Preprocessor(readme_processor, 2).process
    return preprocessor(r.text)

def get_ai_readmes(repos, workers):
    with ThreadPoolExecutor(workers) as executor:
        texts = executor.map(get_readme, repos)
        return [t for t in texts if t]
    
def get_embeddings(filename, size, dims = 100):
    embeddings = np.ones((size, dims))
    with open(filename, 'r') as f:
        j = 0
        for i,l in enumerate(f):
            if i%2 == 1:
                a = np.array([float(n) for n in l.strip().split()])
                embeddings[j,] = a
                j += 1     
            if j == size:
                break
    return embeddings

def get_sentence(i, path):
    return linecache.getline(path, i+1)

def print_random(pos, amt=30, path = './prepared-readmes/repos.txt'):
    print('Printing {}/{} positive documents'.format(amt, len(pos)))
    np.random.shuffle(pos)
    for p in pos[0:30]: 
        print(get_sentence(p[0], path))

def get_ss_embed(out):
    arr = out.split('\n')[4:]
    vecs = [np.array([float(n) for n in s.strip().split(' ')]) 
            for i,s in enumerate(arr) if i%2 == 1] 
    return vecs

def get_repo_name(i):
    """Get id from lookup... then get name from bigquery table? """
    pass
    
def predict_ai(model, pos, X = None):
    """ pos is the positive class vectors, X is the entire dataset """    
    model = deepcopy(model)
    model.fit(pos)
    if X is None:
        return model
    preds = model.predict(X)
    return np.argwhere(preds == 1), model
        
def write_predictions(pos, filename):
    with open(filename, 'w') as f:
        for p in pos.reshape(-1):
            f.write('{}\n'.format(p))

def get_predictions(filename):
    with open(filename, 'r') as f:
        return [int(i) for i in f.read().split('\n') if i]
        
