from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from surprise import KNNBaseline, Reader
from surprise import Dataset

import os
import io
import pickle

# rebuild the map dictionary from playlist_id to playlist name
id_name_dic = pickle.load(open("popular_playlist.pkl", "rb"))
print("loading the map dictionary from playlist_id to playlist name...")

# rebuild the map dictionary from playlist_id to playlist name
name_id_dic = {}
for playlist_id in id_name_dic:
    name_id_dic[id_name_dic[playlist_id]] = playlist_id
print("loading the map dictionary from playlist name to playlist_id...")

file_path = os.path.expanduser('./popular_music_surprise_format.txt')
# assign the file format
reader = Reader(line_format='user item rating timestamp', sep=',')
# get the data from the file
music_data = Dataset.load_from_file(file_path, reader=reader)
# calculate the similarity from songs to songs
print("building the data set...")
trainset = music_data.build_full_trainset()
# sim_options = {'name': 'pearson_baseline', 'user_based': False}


# find the nearest playlist
print("start to train model...")
# sim_option = {'user_based': False}
# algo = KNNBaseline(sim_option=sim_option)
algo = KNNBaseline
algo.train(trainset)

current_playlist = name_id_dic.keys()[39]
print("playlist name:", current_playlist)

""" predict to the playlist"""
# get the nearest neighbors
#  map name to id
playlist_id = name_id_dic[current_playlist]
print("playlist_id:", playlist_id)
# get the inner user id -> to_inner_uid
playlist_inner_id = algo.trainset.to_inner_uid(playlist_id)
print("inner id:", playlist_inner_id)

playlist_neighbors = algo.get_neighbors(playlist_inner_id, k=10)

# put the playlist_id into playlist name
# to_raw_id used to map back
playlist_neighbors = (algo.trainset.to_raw_uid(inner_id)
                      for inner_id in playlist_neighbors)
playlist_neighbors = (id_name_dic[playlist_id]
                      for playlist_id in playlist_neighbors)
print()
print("the nearest 10 play lists with <<", current_playlist, ">> is: \n")
for playlist in playlist_neighbors:
    print(playlist, algo.trainset.to_inner_uid(name_id_dic[playlist]))

""" predict to the users"""
# rebuild the map dictionary from song_id to song name
song_id_name_dic = pickle.load(open("popular_song.pkl", "rb"))
print("loading the map dictionary from song_id to song name...")

# rebuild the map dictionary from playlist_id to playlist name
song_name_id_dic = {}
for song_id in song_id_name_dic:
    song_name_id_dic[song_id_name_dic[song_id]] = song_id
print("loading the map dictionary from song name to song_id...")

# just take the user id is 4
user_inner_id = 4
user_rating = trainset.ur[user_inner_id]
items = map(lambda x: x[0], user_rating)
for song in items:
    print(algo.predict(user_inner_id, song, r_ui=1),
          song_id_name_dic[algo.trainset.to_raw_iid(song)])

