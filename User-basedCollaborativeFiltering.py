# coding : utf-8
# use pySpark to realize the user-based collaborative filter
# use Cosine Similarity

import sys
import random
import pdb
import numpy as np

from collections import defaultdict
from itertools import combinations
from pyspark import SparkContext


# user, item, rating, timestamp
def parse_vector_on_user(line):
    """
    analysis data, key is user, afterward is item and score
    """

    line = line.split("|")
    return line[0], (line[1], float(line[2]))


def parse_vector_on_item(line):
    """
    analysis data, key is item, afterward is user and score
    """
    line = line.split("|")
    return line[1], (line[0], float(line[2]))


def sample_interactions(item_id, users_with_rating, n):
    """
    if some items have extremely much user behavior, could make some down sample appropriately
    """
    if len(users_with_rating) > n:  # we could set n in any number
        return item_id, random.sample(users_with_rating, n)
    else:
        return item_id, users_with_rating


def find_user_pairs(item_id, users_with_rating):
    """
    to every item, find the same user pair to make score
    """
    for user1, user2 in combinations(users_with_rating, 2):
        return (user1[0], user2[0]), (user1[1], user2[1])


def cosine(dot_product, rating1_norm_squared, rating2_norm_squared):
    """
    the cosine similarity of two vector A and B
    dot_product(A, B) / (norm(A) * norm(B))
    """
    numerator = dot_product
    denominator = rating1_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0


def calc_sim(user_pair, rating_pairs):
    """
    to every user pairs, according to scores to calculate the cosine distance, and return the common number of items
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
    return user_pair, (cos_sim, n)


def key_of_first_user(user_pair, item_sim_data):
    """
    to every user-user pairs, use the first one to be the key
    """
    (user1_id, user2_id) = user_pair
    return user1_id, (user2_id, item_sim_data)


def nearest_neighbors(user, user_and_sims, n):
    """
    pick up the nearest N neighbors
    """
    user_and_sims.sort(key=lambda x: x[1][0], reversed=True)
    return user, user_and_sims[:n]


def topN_Recommendations(user_id, user_sims, user_with_rating, n):
    """
    according to the nearest N neighbors to recommend
    """
    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for(neighbor, (sim, count)) in user_sims:
        # traverse neighbors' scores
        unscored_items = user_with_rating.get(neighbor, None)

        if unscored_items:
            for(item, rating) in unscored_items:
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

    sc = SparkContext(sys.argv[1], "PythonUserCF")
    lines = sc.textFile(sys.argv[2])
    """
    process the data, get sparse item-user matrix:
    item_id -> ((user1, rating), (user2, rating))
    """
    item_user_pairs = lines.map(parse_vector_on_item).groupByKey().map(
        lambda p: sample_interactions(p[0], p[1], 500)).cache()
    """
    get two users item-item pairs scores combination:
    (user1_id, user2_id)->[(rating1, rating2),
                           (rating1, rating2),
                           (rating1, rating2)
                           ...]
    """
    pairwise_user = item_user_pairs.filter(
        lambda p: len(p[1]) > 1).map(
        lambda p: find_user_pairs(p[0], p[1])).groupByKey()
    """
    calculate th cosine similarity and find nearest N neighbors:
    (user1, user2)->(similarity, co_raters_count)
    """
    user_sims = pairwise_user.map(
        lambda p: calc_sim(p[0], p[1])).map(
        lambda p: key_of_first_user(p[0], p[1])).groupByKey().map(
        lambda p: nearest_neighbors(p[0], p[1], 50))
    """
    to every user's score recording transfer to this:
    user_id -> [(item_id_1, rating_1),
                (item_id_2, rating_2),
                ...]
    """
    user_item_hist = lines.map(parse_vector_on_user).groupByKey().collect()
    ui_dict = {}
    for(user, items) in user_item_hist:
        ui_dict[user] = items

    uib = sc.broadcast(ui_dict)
    """
    calculate Top N to all users
    user_id -> [item1, item2, item3,...]
    """
    user_item_recs = user_sims.map(lambda p: topN_Recommendations(p[0], p[1], uib.value, 100)).collect()

