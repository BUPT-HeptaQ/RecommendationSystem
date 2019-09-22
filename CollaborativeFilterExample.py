"""
According to item to feed back the highest similarity item
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from surprise import KNNBaseline
from surprise import Dataset

import os
import io


def read_item_names():
    """
    get the movie name and movie id and the map of movie id to movie name
    """
    file_name = (os.path.expanduser('~') + 'u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


# first use algorithm to calculate the similarity between each other
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
sim_options = {'names': 'pearson_baseline', 'user_based': False}  # hyper-parameter
algo = KNNBaseline(sim_options=sim_options)

# get the map of movie name to movie_id and movie_id to movie name
rid_to_name, name_to_rid = read_item_names()

# get the <<Toy Story>> and its item_id
toy_story_raw_id = name_to_rid['Toy Story (1995)']
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

# find 10 nearest neighbors
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

# from the nearest id map back the movie name
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in toy_story_neighbors)
toy_story_neighbors = (rid_to_name[rid]
                       for rid in toy_story_neighbors)

print('The 10 nearest neighbors of Toy Story are:')
for movie in toy_story_neighbors:
    print(movie)

