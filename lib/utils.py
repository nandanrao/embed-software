import requests
import numpy as np
import linecache

def get_readme(repo, attempts = 0):
    filenames = [ 'README.md', 'readme.md']
    try: 
        f = filenames[attempts]
    except IndexError:
        return None
    r = requests.get('https://raw.githubusercontent.com/{}/master/{}'.format(repo, f))
    if r.status_code == 404:
        return get_readme(repo, attempts + 1)
    return preprocessor(r.text)

def write_ai_readmes(repos, filename):
    with open(filename, 'w') as f:
        for repo in repos:
            text = get_readme(repo)
            if text:
                f.write(text + '\n')

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

def get_ss_embed(filename):
    with open(filename, 'r') as f:
        foo = [l.strip().split(' ') for i,l in enumerate(f)]
        vecs = [np.array([float(n) for n in s]) 
                for i,s in enumerate(foo) if i%2 == 1]
        return vecs
