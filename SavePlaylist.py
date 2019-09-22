# coding: utf-8
# save playlist_id->playlist name and song_id->song name

import pickle


def parse_playlist_get_info(in_line, playlist_dic, song_dic):
    contents = in_line.strip().split("\t")
    name, tags, playlist_id, subscribed_count = contents[0].split("##")
    playlist_dic[playlist_id] = name
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split("::")
            song_dic[song_id] = song_name + "\t" + artist

        except (OSError, TypeError) as reason:
            print("the error info is:", str(reason))
            print("song format error")
            print(song + "\n")
            continue


def parse_file(in_file, out_playlist, out_song):
    # the map dictionary from playlist_id to playlist name
    playlist_dic = {}
    # the map dictionary from song_id to song name
    song_dic = {}
    for line in open(in_file):
        parse_playlist_get_info(line, playlist_dic, song_dic)
    # push the map dictionary into binary dictionary files
    pickle.dump(playlist_dic, open(out_playlist, "wb"))
    # could use playlist_dic = pickle.load(open("playlist.plk", "rb")) to reload
    pickle.dump(song_dic, open(out_song, "wb"))


parse_file(".txt", "playlist.pkl", "song.pkl")

