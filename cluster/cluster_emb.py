from sklearn.cluster import MiniBatchKMeans
import numpy as np
from helper import features
import pandas as pd
import time

FOLDER = '../data_split01/'
    
def create_cluster_overlap( combi, folder=FOLDER, size=100, esize=20, prefix='nmf150' ):
    
    start = time.time()
    
    UF = ['uf_'+str(i) for i in range(esize)]
    
    fname = prefix + '_user_features.' + str(esize) + '.csv'
    user_emb = pd.read_csv( folder + fname )
    
    train = combi[combi.train==1].msno.unique()
    del combi
    
    cluster = user_emb[ np.in1d( user_emb.msno, train ) ]
    rest = user_emb[ np.in1d( user_emb.msno, train, invert=True ) ]
    
    SPMC = cluster[UF]
    SPMR = rest[UF]
    
    print( 'created user features in ',(time.time() - start) )
    
    start = time.time()
     
    mbkm = MiniBatchKMeans( max_iter=500, verbose=1, n_clusters=int( user_emb.msno.nunique() / size ) )
    mbkm.fit( SPMC )
    
    usrA = pd.DataFrame()
    usrA['msno'] = cluster.msno
    usrA['cluster_id'] = mbkm.predict( SPMC )
    
    print( 'created KM cluster features in ',(time.time() - start) )
    print( 'KM-CLUSTER: ',int( user_emb.msno.nunique() / size ) )
        
    usrB = pd.DataFrame()
    usrB['msno'] = rest.msno.unique()
    usrB['cluster_id2'] = mbkm.predict( SPMR )
    
    print( 'predicted cluster features in ',(time.time() - start) )
    
    usr = pd.DataFrame()
    usr['msno'] = user_emb.msno.unique()
    usr = usr.merge( usrA, on='msno', how='left' )
    usr = usr.merge( usrB, on='msno', how='left' )
    usr = usr.fillna(0)
    usr['cluster_id'] =  usr['cluster_id'] +  usr['cluster_id2']
    usr = usr[['msno','cluster_id']]
    
    usr.to_csv( folder + prefix+'clusterEMB'+str(esize)+'_msno_ol.'+str(size)+'.csv', index=False )

def create_cluster_song_overlap( combi, folder=FOLDER, size=100, esize=20, prefix='nmf150'):
    
    start = time.time()
    
    SF = ['sf_'+str(i) for i in range(esize)]
    
    fname = prefix + '_song_features.' + str(esize) + '.csv'
    song_emb = pd.read_csv( folder + fname )
    
    train = combi[combi.train==1].song_id.unique()
    del combi
    
    cluster = song_emb[ np.in1d( song_emb.song_id, train ) ]
    rest = song_emb[ np.in1d( song_emb.song_id, train, invert=True ) ]
    
    SPMC = cluster[SF]
    SPMR = rest[SF]
    
    print( 'created song features in ',(time.time() - start) )
    
    start = time.time()
 
    mbkm = MiniBatchKMeans( max_iter=500, verbose=1, n_clusters=int( song_emb.song_id.nunique() / size ) )
    
    sngA = pd.DataFrame()
    sngA['song_id'] = cluster.song_id.unique()
    sngA['cluster_id'] = mbkm.fit_predict( SPMC )
    
    print( 'created KM cluster features in ',(time.time() - start) )
    print( 'KM-CLUSTER: ',int( song_emb.song_id.nunique() / size ) )
        
    sngB = pd.DataFrame()
    sngB['song_id'] = rest.song_id.unique()
    sngB['cluster_id2'] = mbkm.predict( SPMR )
    
    print( 'predicted cluster features in ',(time.time() - start) )
    
    sng = pd.DataFrame()
    sng['song_id'] = song_emb.song_id.unique()
    sng = sng.merge( sngA, on='song_id', how='left' )
    sng = sng.merge( sngB, on='song_id', how='left' )
    sng = sng.fillna(0)
    sng['cluster_id'] =  sng['cluster_id'] +  sng['cluster_id2']
    sng = sng[['song_id','cluster_id']]
    
    sng.to_csv( folder + prefix+'clusterEMB'+str(esize)+'_song_id_ol.'+str(size)+'.csv', index=False )


def create_cluster_artist_overlap( combi, folder=FOLDER, size=100, esize=20, prefix='nmf150'):
    
    start = time.time()
    
    AF = ['af_'+str(i) for i in range(esize)]
    
    fname = prefix + '_artist_features.' + str(esize) + '.csv'
    song_emb = pd.read_csv( folder + fname )
    
    train = combi[combi.train==1].artist_name.unique()
    del combi
    
    cluster = song_emb[ np.in1d( song_emb.artist_name, train ) ]
    rest = song_emb[ np.in1d( song_emb.artist_name, train, invert=True ) ]
    
    SPMC = cluster[AF]
    SPMR = rest[AF]
    
    print( 'created song features in ',(time.time() - start) )
    
    start = time.time()
    
    from sklearn.cluster import MiniBatchKMeans
 
    mbkm = MiniBatchKMeans( max_iter=500, verbose=1, n_clusters=int( song_emb.artist_name.nunique() / size ) )
    
    sngA = pd.DataFrame()
    sngA['artist_name'] = cluster.artist_name.unique()
    sngA['cluster_id'] = mbkm.fit_predict( SPMC )
    
    print( 'created KM cluster features in ',(time.time() - start) )
    print( 'KM-CLUSTER: ',int( song_emb.artist_name.nunique() / size ) )
        
    sngB = pd.DataFrame()
    sngB['artist_name'] = rest.artist_name.unique()
    sngB['cluster_id2'] = mbkm.predict( SPMR )
    
    print( 'predicted cluster features in ',(time.time() - start) )
    
    sng = pd.DataFrame()
    sng['artist_name'] = song_emb.artist_name.unique()
    sng = sng.merge( sngA, on='artist_name', how='left' )
    sng = sng.merge( sngB, on='artist_name', how='left' )
    sng = sng.fillna(0)
    sng['cluster_id'] =  sng['cluster_id'] +  sng['cluster_id2']
    sng = sng[['artist_name','cluster_id']]
    
    sng.to_csv( folder + prefix+'clusterEMB'+str(esize)+'_artist_name_ol.'+str(size)+'.csv', index=False )

def create_cluster_both_overlap( combi, folder=FOLDER, size=100, esize=20, prefix='nmf150'):
    
    start = time.time()
    
    SF = ['sf_'+str(i) for i in range(esize)]
    
    fname = prefix + '_song_features.' + str(esize) + '.csv'
    song_emb = pd.read_csv( folder + fname )
    
    UF = ['uf_'+str(i) for i in range(esize)]
    
    fname = prefix + '_user_features.' + str(esize) + '.csv'
    user_emb = pd.read_csv( folder + fname )
    
    print( user_emb )
    print( song_emb )
    
    combi = combi.merge( user_emb, on='msno', how='inner' )
    combi = combi.merge( song_emb, on='song_id', how='inner' )
    
    combi['u_s'] = combi['msno'].map(str) + '_' + combi['song_id'].map(str)
    combi = combi.drop_duplicates( subset=['u_s'], keep='first' )
    train = combi[combi.train==1].u_s.unique()
    
    cluster = combi[ np.in1d( combi.u_s, train ) ]
    rest = combi[ np.in1d( combi.u_s, train, invert=True ) ]
    
    SPMC = cluster[SF+UF]
    SPMR = rest[SF+UF]
    
    print( 'created song features in ',(time.time() - start) )
    
    start = time.time()
     
    mbkm = MiniBatchKMeans( max_iter=500, verbose=1, n_clusters=int( song_emb.song_id.nunique() / size ) )
    
    sngA = pd.DataFrame()
    sngA['u_s'] = cluster.u_s.unique()
    sngA['cluster_id'] = mbkm.fit_predict( SPMC )
    
    print( 'created KM cluster features in ',(time.time() - start) )
    print( 'KM-CLUSTER: ',int( song_emb.song_id.nunique() / size ) )
        
    sngB = pd.DataFrame()
    sngB['u_s'] = rest.u_s.unique()
    sngB['cluster_id2'] = mbkm.predict( SPMR )
    
    print( 'predicted cluster features in ',(time.time() - start) )
    
    sng = combi[['msno','song_id','u_s']]
    sng = sng.merge( sngA, on='u_s', how='left' )
    sng = sng.merge( sngB, on='u_s', how='left' )
    sng = sng.fillna(0)
    sng['cluster_id'] =  sng['cluster_id'] +  sng['cluster_id2']
    sng = sng[['msno','song_id','cluster_id']]
    
    sng.to_csv( folder + prefix+'clusterEMB'+str(esize)+'_both_ol.'+str(size)+'.csv', index=False )

if __name__ == '__main__':
    
    combi = features.load_combi_prep(FOLDER)
    #create_cluster(combi, size=10)
    #create_song_cluster(combi, size=100, pos=False)
    
    create_cluster_overlap(combi, size=25, esize=32, prefix='als')
    create_cluster_song_overlap(combi, size=25, esize=32, prefix='als')
    create_cluster_artist_overlap(combi, size=25, esize=32, prefix='als')
    create_cluster_both_overlap(combi, size=25, esize=32, prefix='als')
    