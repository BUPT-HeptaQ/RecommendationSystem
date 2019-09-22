# coding : utf-8
# use pySpark to realize the item-based collaborative filter


import sys
import random
import csv
import pdb
import numpy as np

from math import sqrt
from collections import defaultdict
from itertools import combinations
from pyspark import SparkContext


def parse_vector(line):
    """
    analysis data, key is item, afterward is user and score
    """
    line = line.split("|")
    return line[0], (line[1], float(line[2]))


def sample_interactions(user_id, items_with_rating, n):
    """
    if some users have extremely much score behavior, could make some down sample appropriately
    """
    if len(items_with_rating) > n:
        return user_id, random.sample(items_with_rating, n)
    else:
        return user_id, items_with_rating


def find_item_pairs(user_id, items_with_rating):
    """
    pair to every user's score and item
    """
    for item1, item2 in combinations(items_with_rating, 2):
        return (item1[1], item2[0]), (item1[1], item2[1])


def cosine(dot_product, rating1_norm_squared, rating2_norm_squared):
    """
    the cosine similarity between two vectors A and B
    dot_product(A, B) / (norm(A) * norm(B))
    """
    numerator = dot_product
    denominator = rating1_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0


def calc_sim(item_pair, rating_pairs):
    """
    to every item pairs, according to scores to calculate the cosine distance, and return the common number of users
    """
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)

    for rating_pairs in rating_pairs:
        sum_xx += np.float(rating_pairs[0]) * np.float(rating_pairs[0])
        sum_yy += np.float(rating_pairs[1]) * np.float(rating_pairs[1])
        sum_xy += np.float(rating_pairs[0]) * np.float(rating_pairs[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy, np.sqrt(sum_xx), np.sqrt(sum_yy))
    return item_pair, (cos_sim, n)


def correlation(size, dot_product, rating_sum,
                rating2sum, rating1_norm_squared, rating2_norm_squared):
    """
     the similarity between two vectors A and B
        [n * dotProduct(A, B) - sum(A) * sum(B)] /
        sqrt { [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }
    """
    numerator = size * dot_product - rating_sum * rating2sum
    denominator1 = sqrt(size * rating1_norm_squared - rating_sum * rating_sum)
    denominator2 = sqrt(size * rating2_norm_squared - rating2sum * rating2sum)
    denominator_product = denominator1 * denominator2

    return (numerator / (float(denominator_product))) if denominator_product else 0.0


def key_of_first_item(item_pair, item_sim_data):
    """
    to every item-item pairs, use the first one to be the key
    """
    (item1_id, item2_id) = item_pair
    return item1_id, (item2_id, item_sim_data)


def nearest_neighbors(item_id, items_and_sims, n):
    """
    pick up the nearest N neighbors
    """
    items_and_sims.sort(key=lambda x: x[1][0], reversed=True)
    return item_id, items_and_sims[:n]


def topN_Recommendations(user_id, items_with_rating, item_sims, n):
    """
    according to the nearest N neighbors to recommend
    """
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for(item, rating) in items_with_rating:

        # traverse items' neighbors
        nearest_neighbors = item_sims.get(item, None)

        if nearest_neighbors:
            for(neighbor, (sim, count)) in nearest_neighbors:
                if neighbor != item:

                    # update the recommendation and similarity
                    totals[neighbor] += sim*rating
                    sim_sums[neighbor] += sim

    # normalization
    scored_items = [(total / sim_sums[item], item) for item, total in totals.items()]

    # according to the strength of recommendations get descending sort
    scored_items.sort(reverse=True)

    # The item's strength of recommendations
    ranked_items = [x[1] for x in scored_items]

    return user_id, ranked_items[:n]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: PythonUserCF <master> <file>", sys.stderr)
        exit(-1)

    sc = SparkContext(sys.argv[1], "PythonItemCF")
    lines = sc.textFile(sys.argv[2])
    """
    process the data, get sparse user-item matrix:
    user_id -> [(item_id_1, rating_1), 
                (item_id_2, rating_2),
                ...]
    """
    user_item_pairs = lines.map(parse_vector).groupByKey().map(
        lambda p: sample_interactions(p[0], p[1], 500)).cache()
    """
    get all item-item pairs combination:
    (item1, item2)->[(item1_rating, item2_rating),
                     (item1_rating, item2_rating),
                     ...]
    """
    pairwise_items = user_item_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: find_item_pairs(p[0], p[1])).groupByKey()
    """
    calculate th cosine similarity and find nearest N neighbors:
    (item1, item2)->(similarity, co_raters_count)
    """
    item_sims = pairwise_items.map(
        lambda p: calc_sim(p[0], p[1])).map(
        lambda p: key_of_first_item(p[0], p[1])).groupByKey().map(
        lambda p: nearest_neighbors(p[0], p[1], 50)).collect()

    item_sim_dict = {}
    for(item, data) in item_sims:
        item_sim_dict[item] = data

    isb = sc.broadcast(item_sim_dict)
    """
    calculate Top N recommendation outcomes 
    user_id -> [item1, item2, item3,...]
    """
    user_item_recs = user_item_pairs.map(lambda p: topN_Recommendations(p[0], p[1], isb.value, 500)).collect()

