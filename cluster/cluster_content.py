from sklearn.cluster import MiniBatchKMeans
import numpy as np
from helper import features
import pandas as pd
import time
import helper.features as f

FOLDER = '../data/'

CONTENT_USER = ['msno','bd', 'city','expiration_date','expiration_day', 'expiration_month', 'expiration_year', 'gender','membership_days', 'membership_diff','registered_via',
       'registration_day', 'registration_date', 'registration_month',
       'registration_year']

CONTENT_ITEM = ['song_id','artist_name', 'artist_name_first', 'artist_name_max','composer', 'composer_first', 'composer_max',
       'genre_first', 'genre_ids', 'genre_max','language','lyricist',
       'lyricist_first', 'lyricist_max','song_length','tags_mb_first', 'tags_mb_max']

MCOL_FEATURES = ['source_screen_name','source_system_tab', 'source_type']
MCOL_FEATURES_MAX = ['source_screen_name_max','source_system_tab_max', 'source_type_max','u_played','u_played_rel']

S_MCOL_FEATURES = ['source_screen_name','source_system_tab', 'source_type']
S_MCOL_FEATURES_MAX = ['source_screen_name_max','source_system_tab_max', 'source_type_max','s_played','s_played_rel','a_played','a_played_rel']

def create_cluster_overlap( combi, folder=FOLDER, size=100, pos=True ):
    
    start = time.time()
    
    train_user = combi[combi.train==1].msno.unique()
        
    combi = mcol_max_features(combi, 'msno', MCOL_FEATURES)
    combi = f.scol_features( combi, 'msno', 'u_' )
    
    cluster = combi[ np.in1d( combi.msno, train_user ) ]
    rest = combi[ np.in1d( combi.msno, train_user, invert=True ) ]
    
    SPMC = cluster[CONTENT_USER+MCOL_FEATURES_MAX].groupby('msno').max()
    SPMR = rest[CONTENT_USER+MCOL_FEATURES_MAX].groupby('msno').max()
    
    print( 'created user features in ',(time.time() - start) )
    
    start = time.time()
     
    mbkm = MiniBatchKMeans( verbose=1, max_iter=500, n_clusters=int( combi.msno.nunique() / size ) )
    
    usrA = pd.DataFrame()
    usrA['msno'] = cluster.msno.unique()
    mbkm.fit( SPMC )
    usrA['cluster_id'] = mbkm.predict( SPMC )
    
    print( 'created KM cluster features in ',(time.time() - start) )
    print( 'KM-CLUSTER: ',int( combi.msno.nunique() / size ) )
        
    usrB = pd.DataFrame()
    usrB['msno'] = rest.msno.unique()
    usrB['cluster_id2'] = mbkm.predict( SPMR )
    
    print( 'predicted cluster features in ',(time.time() - start) )
    
    usr = pd.DataFrame()
    usr['msno'] = combi.msno.unique()
    usr = usr.merge( usrA, on='msno', how='left' )
    usr = usr.merge( usrB, on='msno', how='left' )
    usr = usr.fillna(0)
    usr['cluster_id'] =  usr['cluster_id'] +  usr['cluster_id2']
    usr = usr[['msno','cluster_id']]
    
    usr.to_csv( folder + 'content_cluster_msno_ol'+('_nopos' if not pos else '')+'.'+str(size)+'.csv', index=False )

def create_cluster_song_overlap( combi, folder=FOLDER, size=100, pos=True ):
    
    start = time.time()
    
    train_songs = combi[combi.train==1].song_id.unique()
        
    combi = mcol_max_features(combi, 'song_id', MCOL_FEATURES)
    combi = f.scol_features( combi, 'song_id', 's_' )
    combi = f.scol_features( combi, 'artist_name', 'a_' )
    
    combi = combi.dropna(axis=1)
    
    cluster = combi[ np.in1d( combi.song_id, train_songs ) ]
    rest = combi[ np.in1d( combi.song_id, train_songs, invert=True ) ]
    
    SPMC = cluster[CONTENT_ITEM+S_MCOL_FEATURES_MAX].groupby('song_id').max()
    SPMR = rest[CONTENT_ITEM+S_MCOL_FEATURES_MAX].groupby('song_id').max()
    
    print( 'created song features in ',(time.time() - start) )
    
    start = time.time()
 
    mbkm = MiniBatchKMeans( verbose=1, max_iter=500, n_clusters=int( combi.song_id.nunique() / size ) )
    
    sngA = pd.DataFrame()
    sngA['song_id'] = cluster.song_id.unique()
    mbkm.fit( SPMC )
    sngA['cluster_id'] = mbkm.predict( SPMC )
    
    print( 'created KM cluster features in ',(time.time() - start) )
    print( 'KM-CLUSTER: ',int( combi.song_id.nunique() / size ) )
        
    sngB = pd.DataFrame()
    sngB['song_id'] = rest.song_id.unique()
    sngB['cluster_id2'] = mbkm.predict( SPMR )
    
    print( 'predicted cluster features in ',(time.time() - start) )
    
    sng = pd.DataFrame()
    sng['song_id'] = combi.song_id.unique()
    sng = sng.merge( sngA, on='song_id', how='left' )
    sng = sng.merge( sngB, on='song_id', how='left' )
    sng = sng.fillna(0)
    sng['cluster_id'] =  sng['cluster_id'] +  sng['cluster_id2']
    sng = sng[['song_id','cluster_id']]
    
    sng.to_csv( folder + 'content_cluster_song_id_ol'+('_nopos' if not pos else '')+'.'+str(size)+'.csv', index=False )

def mcol_max_features( combi, col, dest ):
    
    start = time.time()
    
    for dcol in dest: 
        
        tmp = pd.DataFrame()
        tmp[dcol+'_max'] = combi.groupby([col,dcol]).size()
        tmp = tmp.reset_index()
        
        combi = pd.merge( combi, tmp, how='inner', on=[col,dcol] )                
        
    print( (','.join(col)), ' features in: ', (time.time() - start) )

    return combi

if __name__ == '__main__':
    
    combi = features.load_combi_prep(FOLDER)
    
    create_cluster_overlap(combi, size=25, pos=False)
    create_cluster_song_overlap(combi, size=25, pos=False)
    