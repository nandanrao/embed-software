#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:19:38 2018

@author: davidrosenfeld
"""

import pandas as pd
import numpy as np
import json, time, os, logging, pprint
from pymongo import MongoClient
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, lil_matrix
from scipy.optimize import brute
from scipy.special import expit
from sklearn.metrics import fbeta_score
from scipy.stats import gamma


# Hand-labelled papers to classify.
# Add more negative papers to make model more conservative!!
POSITIVE_PAPERS = [
    "Gaussian Processes for Big Data",
    "NewsWeeder: Learning to Filter Netnews",
    "Object Detection with Grammar Models",
    "Dropout Training as Adaptive Regularization",
    "Large-scale L-BFGS using MapReduce",
    "Monte-Carlo Planning in Large POMDPs",
    "The Infinite Gaussian Mixture Model",
    "Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks",
    "DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition",
    "Two-Stream Convolutional Networks for Action Recognition in Videos",
    "Dueling Network Architectures for Deep Reinforcement Learning",
    "Auxiliary Deep Generative Models",
    "Boosting for transfer learning",
    "A Neural Probabilistic Language Model",
    "Maximum Margin Temporal Clustering",
    "Learning to rank using gradient descent"
]
NEGATIVE_PAPERS = [
    "How to Train Neural Networks",
    "kernel bayes' rule",
    "Self-taught clustering",
    "real-time particle filters",
    "How Neural Nets Work",
    "Direct and Indirect Effects",
    "Separating Style and Content",
    "Management of Uncertainty",
    "Adaptive Online Learning",
    "Learning Model Bias",
    "Latent Bandits",
    "Noisy Activation Functions",
    "Leaving the Span",
    "Bayesian PCA",
    "Learning Rotations",
    "Robustness and Generalization"
]

# Connect to MongoDB
def connect_to_db():
    client = MongoClient(os.getenv('MONGO_HOST'),
                     username = os.getenv('MONGO_USER'),
                     password = os.getenv('MONGO_PASSWORD'))
    db = client.githubresults
    collection = db.dblp_returns
    return collection

# Function adds or appends a new title as a value given a repo as key
def add_repo_to_dict(repo_dict, repo, title):
    if repo in repo_dict:
        repo_dict[repo].append(title)
    else:
        repo_dict[repo] = [title]
    return repo_dict

# Adds the repo (key) and titles (values) for all titles associated to a repo
def add_repos_to_dict(repo_dict, repo_list, title):
    for repo in repo_list:
        repo_dict = add_repo_to_dict(repo_dict, repo, title)
    return repo_dict

# Creates 3 outputs:
## A dictionary with repos as keys, and values are a list of titles
## A Dictionary with titles as keys, and values are a list of repos
## A list of titles for which the answer was an error because query limit exceeded
def create_repo_dict(all_responses):
    repo_dict = {}
    title_dict = {}
    error_queries = []
    for response in all_responses:
        if "total_count" in response["response"]:
            has_items = response["response"]["total_count"]
            if has_items > 0:
                title = response["_id"]
                items = response["response"]["items"]
                repo_list = [repo["full_name"] for repo in items]
                repo_dict = add_repos_to_dict(repo_dict, repo_list, title)
                title_dict[title] = has_items
        else:
            error_queries.append(response["_id"])
    return repo_dict, title_dict, error_queries

# Subset a dictionary for keys whose values are a list of length higher than min_length
def get_lengthy_keys(dictionary, min_length):
    len_dict = {key: len(value) for key, value in dictionary.items()}
    greedy_keys = [[key, len_dict[key]] for key in len_dict if len_dict[key] > min_length]
    return greedy_keys


# Concatenate strings
def concat_titles(title):
    if len(title) == 1:
        title = title[0]
    else:
        title = ' '.join(title)
    return title

# Function returns repos which have:
### A max_paper_per_repo number of papers cited;
### Excludes papers whose title short_word_limit number of words or fewer,
    # and have more than max_repo_per_title number of repos which cite them.
def get_dicts():
    collection = connect_to_db()
    all_responses = collection.find({})
    repo_dict, title_dict, error_queries = create_repo_dict(all_responses)
    return repo_dict, title_dict


def filter_repeats(l):
    """ Filters case-insensitive repeats from list, taking last element """
    lowered = [i.lower() for i in l]
    uniques = []
    for i,s in enumerate(l):
        idx = [j for j,f in enumerate(lowered)
               if s.lower() == f]
        uniques.append(idx[-1])
    return [s for i,s in enumerate(l) if i in uniques]


def get_sparse_matrix(repo_dict, titles):
    # Just boilerplate CSR building
    indptr = [0]
    indices = []
    data = []
    for papers in repo_dict.values():
        for title in papers:
            index = np.argwhere(titles == title)[0][0]
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr)).tocsc()


def test_df():
    """ Just makes a dataframe with our hand-labelled papers """
    df = pd.DataFrame({'title': POSITIVE_PAPERS + NEGATIVE_PAPERS, 'label': 1})
    df.loc[len(POSITIVE_PAPERS):,'label'] = 0
    return df

# Helper functions for sparse matrix summing
row_sum = lambda m: np.asarray(m.sum(1)).reshape(-1)
col_sum = lambda m: np.asarray(m.sum(0)).reshape(-1)

class TitleClassifier:
    """ Uses word count with Gamma prior over repo count to classify papers """
    def __init__(self, vectorizer = CountVectorizer(), beta = 0.25):
        self.vectorizer = vectorizer
        self.beta = beta

    def fit(self, titles, counts, X, y):
        self.titles = titles
        self.counts = counts
        self.vecs = self.vectorizer.fit_transform(self.titles)

        self.X = X
        self.y = y

        params = brute(self._set_and_test, ((0.33, 2.5), (20, 60)), Ns = 20)
        self._set_params(params)

    def _set_params(self, args):
        self.thresh, self.gamma_scale = args

    def _set_and_test(self, args):
        self.thresh, self.gamma_scale = args
        return self._get_loss()

    def _get_loss(self):
        preds = self.predict(self.X)
        return 1 / fbeta_score(self.y, preds, self.beta)

    def predict_proba(self, titles):
        i = [np.argwhere(title == self.titles)[0][0] for title in titles]
        word_count = row_sum(self.vecs[i])
        repo_count = self.counts[i]
        regularization = gamma.pdf(repo_count, 1, 0, self.gamma_scale)
        score = word_count * regularization * 10
        return expit(score - self.thresh)

    def predict(self, titles):
        decision = self.predict_proba(titles)
        return decision > .5


def filter_repos(repo_dict, title_dict, summary_repo_threshold):

    # Helper functions to sum row- and column-wise
    num_papers, num_repos = row_sum, col_sum

    # Make arrays and sparse matrix of values
    repos = np.array(list(repo_dict.keys()))
    titles = np.array(list(title_dict.keys()))
    counts = np.array(list(title_dict.values()))
    m = get_sparse_matrix(repo_dict, titles)

    # initial sanity check to remove summary repos
    idx = num_papers(m) <= summary_repo_threshold
    counts += (num_repos(m[idx,:]) - num_repos(m))
    m, repos = m[idx, :], repos[idx]

    # classify papers as "real titles" or "common phrases"
    df = test_df()
    c = TitleClassifier(beta = .33)
    c.fit(titles, counts, df.title, df.label)
    idx = c.predict(titles)
    m, titles = m[:, idx], titles[idx]
    logging.info('Classified with threshold: {} \
    and Gamma scale: {}'.format(c.thresh, c.gamma_scale))

    # Return repos with at least one paper
    idx = (num_papers(m) > 0)
    return repos[idx], titles


def get_repos_titles(summary_repo_threshold):
    # Connect to the database and get all query responses
    repo_dict, title_dict = get_dicts()
    repo_dict = {k:filter_repeats(v) for k,v in repo_dict.items()}
    final_repos, final_titles = filter_repos(repo_dict,
                                             title_dict,
                                             summary_repo_threshold)
    return final_repos

if __name__ == '__main__':
    from dotenv import load_dotenv

    p = Path(".") / ".env"
    load_dotenv(dotenv_path = p, verbose=True)
    repos = get_repos_titles(12)
    with open('repos.txt', 'w') as f:
        for repo in repos:
            f.write(repo+'\n')
