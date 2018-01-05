from builtins import str

import pandas as pd
import numpy as np
import time
from helper import features

FOLDER = 'data_common/'


def isrc2year(isrc):
    if type(isrc) == str:
        year = int(isrc[5:7])
        return 1900 + year if year > 17 else 2000 + year
    else:
        return np.nan


if __name__ == '__main__':
    
    start = time.time()
    
    songs = pd.read_csv(FOLDER + 'songs_clean.csv')
    songs = features.split_and_max(songs, 'genre_ids', 'genre_max')
    songs = features.split_and_max(songs, 'composer', 'composer_max')
    songs = features.split_and_max(songs, 'lyricist', 'lyricist_max')
    songs = features.split_and_max(songs, 'artist_name', 'artist_name_max')
    
    songs = features.split_and_first(songs, 'genre_ids', 'genre_first')
    songs = features.split_and_first(songs, 'composer', 'composer_first')
    songs = features.split_and_first(songs, 'lyricist', 'lyricist_first')
    songs = features.split_and_first(songs, 'artist_name', 'artist_name_first')
    
    print('split in: ', (time.time() - start))
    
    extra = pd.read_csv(FOLDER + 'song_extra_info.csv')
    # spotify = pd.read_csv( FOLDER + 'extra/spotify.csv' )
    # musicb = pd.read_csv( FOLDER + 'extra/musicbrainz.csv' )
    
    # LOAD MERGE DATA
    extra = extra.merge(songs, on='song_id', how='left')
#     print(len(extra))
#     extra = extra.merge( spotify, on='song_id', how='left' )
#     print(len(extra))
#     extra = extra.merge( musicb, on='song_id', how='left' )
#     print(len(extra))
    
    del songs  # , spotify, musicb
    
    # SPLIT ISRC
    extra['isrc_year'] = extra['isrc'].apply(isrc2year)
    extra['isrc_country'] = extra['isrc'].str.slice(0, 2)
    
    # CREATE ALTERNATIVE SONG ID WITHOUT DUPLICATES
    name_artist = pd.DataFrame()
    name_artist['size'] = extra.groupby(['name', 'artist_name']).size()
    name_artist = name_artist.reset_index()
    name_artist['song_id_new'] = name_artist.index
    extra = extra.merge(name_artist, on=['name', 'artist_name'], how='left') 
    del name_artist
    
    extra.to_csv(FOLDER + 'songs_complete.csv', encoding='utf-8', index=False)
