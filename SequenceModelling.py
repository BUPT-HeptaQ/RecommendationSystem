# coding: utf-8

import multiprocessing
import gensim
import sys
import pickle

from random import shuffle


def parse_playlist_get_sequence(in_line, playlist_sequence):
    song_sequence = []
    contents = in_line.strip().split("\t")
    # analyse playlist sequence
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split("::")
            song_sequence.append(song_id)

        except (OSError, TypeError) as reason:
            print("the error info is:", str(reason))
            print("song format error")
            print(song + "\n")

    for i in range(len(song_sequence)):
        shuffle(song_sequence)
        playlist_sequence.append(song_sequence)


def train_song2vec(in_file, out_file):
    # all playlist sequence
    playlist_sequence = []
    # traverse all playlist
    for line in open(in_file):
        parse_playlist_get_sequence(line, playlist_sequence)
    # use word2vec to train
    cores = multiprocessing.cpu_count()
    print("using all " + str(cores) + "cores")
    print("Training word2vec model...")
    model = gensim.models.Word2Vec(sentences=playlist_sequence, size=150,
                                   min_count=3, window=7, workers=cores)
    print("Saving model...")
    model.save(out_file)


song_sequence_file = ("./popular.playlist")
model_file = "./song2vec.model"
train_song2vec(song_sequence_file, model_file)

song_dic = pickle.load(open("popular_song.pkl", "rb"))
model_str = str("./song2vec.model")
model = gensim.models.Word2Vec.load(model_str)
for song in song_dic.keys()[:10]:
    print("songs and their IDs:", song + song_dic[song])

song_id_list = song_dic.keys()[1000:1500:50]
for song_id in song_id_list:
    result_song_list = model.most_similar(song_id)

print(song_id_list, song_dic[song_id_list])
print("\n similar songs and the similarity is:")
for song in result_song_list:
    print("\t", song_dic[song[0]], song[1] + "\n")


