#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:19:38 2018

@author: davidrosenfeld
"""

import pandas as pd
import numpy as np
import json
import time
import grequests
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pathlib import Path
import pprint

# Connect to MongoDB
def connect_to_db():
    p = Path(".") / ".env"
    load_dotenv(dotenv_path = p, verbose=True)
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
                title_dict[title] = repo_list
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


from scipy.sparse import csr_matrix, lil_matrix

def get_titles(repo_dict):
    u = [i for v in repo_dict.values() for i in v]
    unique_titles = np.array(list(set(u)))
    return unique_titles


def get_sparse_matrix(repo_dict):
    titles = get_titles(repo_dict)

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

    return csr_matrix((data, indices, indptr)).tocsc(), titles


def filter_repos(repo_dict, max_paper_per_repo, paper_filters):

    # Make sparse matrix > repos x titles
    m, titles = get_sparse_matrix(repo_dict)
    repos = np.array(list(repo_dict.keys()))

    # Helper functions to sum row- and column-wise
    num_papers = lambda m: np.asarray(m.sum(1)).reshape(-1)
    num_repos = lambda m: np.asarray(m.sum(0)).reshape(-1)

    # Filter colummns based on filters over titles
    for word_limit, max_repo_per_paper in paper_filters:
        lengths = pd.Series(titles).map(lambda s: len(s.split(' '))).values
        idx = (num_repos(m) <= max_repo_per_paper) | (lengths > word_limit)
        m,titles = m[:,idx], titles[idx]

    # Return repos with correct number of papers, accounting
    # for the removed papers.
    idx = (num_papers(m) > 0) & (num_papers(m) <= max_paper_per_repo)
    return repos[idx], titles


def get_repos_titles(max_paper_per_repo, paper_filters):

    # Connect to the database and get all query responses
    repo_dict, title_dict = get_dicts()
    repo_dict = {k:filter_repeats(v) for k,v in repo_dict.items()}

    final_repos, final_titles = filter_repos(repo_dict, max_paper_per_repo, paper_filters)
    return final_repos


if __name__ == '__main__':
    repos = get_repos_titles(1, [(2,1), (5, 5)])
    with open('repos.txt', 'w') as f:
        for repo in repos:
            f.write(repo+'\n')
