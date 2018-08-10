from filter_responses import *

def test_filter_repeats():
    rep = ['foo bar', 'ok go', 'Foo Bar']
    out = filter_repeats(rep)
    assert(out == ['ok go', 'Foo Bar'])

def test_filter_repeats_on_empty():
    rep = []
    out = filter_repeats(rep)
    assert(out == [])

def test_filter_repeats_same():
    rep = ['foo bar', 'ok go']
    out = filter_repeats(rep)
    assert(out == rep)

def test_filter_repos_basic_nums():
    repo_dict = {
        'foo': ['Paper1'],
        'bar': ['Paper1', 'Paper2'],
        'baz': ['Paper1'],
        'qux': ['Paper3'],
    }
    assert(filter_repos(repo_dict,1,[(1,1)])[0].tolist() == ['bar', 'qux'])


def test_filter_repos_basic_rids_small_repos():
    repo_dict = {
        'foo': ['Paper One'],
        'bar': ['Paper One', 'Paper Two'],
        'baz': ['Paper Three'],
    }
    assert(filter_repos(repo_dict,1,[(2,1)])[0].tolist() == ['bar', 'baz'])

def test_filter_repos_basic_rids_small_repos():
    repo_dict = {
        'foo': ['Paper One One'],
        'bar': ['Paper One One', 'Paper Two'],
        'baz': ['Paper One One'],
    }
    assert(filter_repos(repo_dict,1,[(2,1), (10, 2)])[0].tolist() == ['bar'])
