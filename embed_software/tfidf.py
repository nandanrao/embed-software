from itertools import chain
from lib.utils import * 
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.corpora.textcorpus import TextCorpus


def create_dict(path, save=''):
    corpus = TextCorpus(path)
    d = Dictionary((c for c in corpus.get_texts()))
    if save:
        d.save(save)
    return d

def all_readmes(path):
    with open(path, 'r') as f:
        for l in f:
            yield l.strip()
            
def embed(readmes, gh_dict):
    vectorizer = TfidfVectorizer(stop_words = REPO_STOP_WORDS, vocabulary = gh_dict.token2id)
    docs = chain(all_readmes('prepared-readmes/repos.txt'), readmes)
    tfidf_embeddings = vectorizer.fit_transform(docs)
    return tfidf_embeddings, vectorizer

def make_predictions(model, embeddings, readmes, save=''):
    model.fit(embeddings[-len(readmes):])
    pos = model.predict(embeddings[:-len(readmes)])
    pos = np.argwhere(pos == 1)
    if save:
        write_predictions(pos, save)
    return pos

if __name__ == '__main__':
    gh_dict = Dictionary().load('prepared-readmes/gh-raw-dict')
    
    gh_dict.filter_extremes(no_below=200)
    c = TextCorpus()
    gh_dict.add_documents([c.preprocess_text(r) for r in readmes])
    
    svm = OneClassSVM(kernel='rbf', nu = .6, gamma = 1/1000)
    tfidf_embeddings = tfidf.embed(readmes, gh_dict)

