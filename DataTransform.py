# coding: utf-8
# original data -> play list data

import json
import sys


def parse_song_line(in_line):
    data = json.loads(in_line)
    name = data['result']['name']
    tags = ",".join(data['result']['tags'])
    subscribed_count = data['result']['subscribedCount']
    if (subscribed_count < 100):  # filter the subscribe number > 100
        return False
    playlist_id = data['result']['id']
    song_info = ' '
    songs = data['result']['tracks']
    for song in songs:  # combine all song in play lists
        try:
            song_info += "\t" + "::".join([str(song['id']), song['name'],
                                           song['artists'][0]['name'], str(song['popularity'])])
        except (OSError, TypeError) as reason:
            print("the error info is:", str(reason))
            continue
    return name + "##" + tags + str(playlist_id) + "##" + str(subscribed_count) + song_info


def parse_file(in_file, out_file):
    out = open(out_file, 'w')
    for line in open(in_file):
        result = parse_song_line(line)
        if(result):
            out.write(result.encode('utf-8').strip()+"\n")
    out.close()


parse_file("json", "txt")  # (data resource, output data)

