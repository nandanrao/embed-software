import requests
import numpy as np
import linecache
from lib.preprocess import preprocessor
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce


def get_readme(repo, attempts = 0):
    """ Get readmes from Github and run through our preprocessor """
    names = ['README', 'readme', 'Readme']
    endings = ['', 'md', 'txt', 'markdown']
    filenames = [n+e for n in names for e in endings]
    print(filenames)
    try: 
        f = filenames[attempts]
    except IndexError:
        print('REPO 404: {}'.format(repo))
        return None
    r = requests.get('https://raw.githubusercontent.com/{}/master/{}'.format(repo, f))
    if r.status_code == 404:
        return get_readme(repo, attempts + 1)
    return preprocessor(r.text)

def get_all_readmes(repos):
    with ThreadPoolExecutor(max_workers=100) as executor:
        return executor.map(get_readme, repos)
    
def get_ai_readmes(repos, filename):
    texts = get_all_readmes(repos)
    return reduce(lambda r,t: r + t + '\n' if t else r, texts, '')

def get_embeddings(filename, size):
    embeddings = np.ones((size, 100))
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

def get_sentence(i):
    return linecache.getline('./prepared-readmes/by_repo.txt', i+1)

def get_ss_embed(out):
    arr = out.split('\n')[4:]
    vecs = [np.array([float(n) for n in s.strip().split(' ')]) 
            for i,s in enumerate(arr) if i%2 == 1] 
    return vecs

def get_repo_name(i):
    """Get id from lookup... then get name from bigquery table? """
    pass
    
def predict_ai(model, pos, X):
    """ pos is the positive class vectors, X is the entire dataset """    
    model = deepcopy(model)
    model.fit(pos)
    preds = model.predict(X)
    return np.argwhere(preds == 1), model

def print_random(pos, amt=30):
    print('Printing {}/{} positive documents'.format(amt, len(pos)))
    np.random.shuffle(pos)
    for p in pos[0:30]: 
        print(get_sentence(p[0]))
