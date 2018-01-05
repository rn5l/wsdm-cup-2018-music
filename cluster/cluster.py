'''
Created on 10.10.2017

@author: ludewig
'''
import scipy.sparse as sparse
import numpy as np
from helper import features
import pandas as pd
import time

FOLDER = '../data_split/'
    
def create_cluster_overlap( combi, folder=FOLDER, size=100, pos=True ):
    
    start = time.time()
    
    train = combi[combi.train==1].msno.unique()
    
    combi = combi[ [ 'msno', 'song_id', 'target' ] ]
    combi['value'] = 1
    if pos:
        combi['target'] = combi['target'].fillna( 0.5 )
        combi['value'] = combi['value'] + 20 * combi['target']
    
    cluster = combi[ np.in1d( combi.msno, train ) ]
    rest = combi[ np.in1d( combi.msno, train, invert=True ) ]
    
    SPMC = create_spm( cluster, 'value', cluster.msno.unique(), combi.song_id.unique() )
    SPMR = create_spm( rest, 'value', rest.msno.unique(), combi.song_id.unique() )
    
    print( 'created user features in ',(time.time() - start) )
    
    start = time.time()
    
    from sklearn.cluster import MiniBatchKMeans
 
    mbkm = MiniBatchKMeans(  n_clusters=int( combi.msno.nunique() / size ) )
    
    usrA = pd.DataFrame()
    usrA['msno'] = cluster.msno.unique()
    usrA['cluster_id'] = mbkm.fit_predict( SPMC )
    
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
    
    usr.to_csv( folder + 'cluster20_msno_ol'+('_nopos' if not pos else '')+'.'+str(size)+'.csv', index=False )

def create_cluster_song_overlap( combi, folder=FOLDER, size=100, pos=True ):
    
    start = time.time()
    
    train = combi[combi.train==1].song_id.unique()
    
    combi = combi[ [ 'msno', 'song_id', 'target' ] ]
    combi['value'] = 1
    if pos:
        combi['target'] = combi['target'].fillna( 0.5 )
        combi['value'] = combi['value'] + 20 * combi['target']
    
    cluster = combi[ np.in1d( combi.song_id, train ) ]
    rest = combi[ np.in1d( combi.song_id, train, invert=True ) ]
    
    SPMC = create_spm_song( cluster, 'value', cluster.song_id.unique(), combi.msno.unique() )
    SPMR = create_spm_song( rest, 'value', rest.song_id.unique(), combi.msno.unique() )
    
    print( 'created song features in ',(time.time() - start) )
    
    start = time.time()
    
    from sklearn.cluster import MiniBatchKMeans
 
    mbkm = MiniBatchKMeans(  n_clusters=int( combi.song_id.nunique() / size ) )
    
    sngA = pd.DataFrame()
    sngA['song_id'] = cluster.song_id.unique()
    sngA['cluster_id'] = mbkm.fit_predict( SPMC )
    
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
    
    sng.to_csv( folder + 'cluster20_song_id_ol'+('_nopos' if not pos else '')+'.'+str(size)+'.csv', index=False )

def create_spm(df, column, user, items):
    
    person_u = list(user)
    thing_u = list(items)
     
    data = df[column].tolist()
    row = df.msno.astype('category', categories=person_u).cat.codes
    col = df.song_id.astype('category', categories=thing_u).cat.codes
    return sparse.csr_matrix((data, (row, col)), shape=(len(person_u), len(thing_u)))

def create_spm_song(df, column, items, user):
    
    person_u = list(user)
    thing_u = list(items)
     
    data = df[column].tolist()
    row = df.song_id.astype('category', categories=thing_u).cat.codes
    col = df.msno.astype('category', categories=person_u).cat.codes
    return sparse.csr_matrix((data, (row, col)), shape=(len(thing_u), len(person_u)))

if __name__ == '__main__':
    
    combi = features.load_combi_prep(FOLDER)
    
    create_cluster_overlap(combi, size=25, pos=True)
    create_cluster_song_overlap(combi, size=25, pos=True)
    