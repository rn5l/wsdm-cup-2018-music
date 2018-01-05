from helper import features
import pandas as pd
import numpy as np
import time
from mf.bpr.bpr import BPR

FOLDER = '../data_split05/'

def create_latent_factors( combi, folder=FOLDER, size=10, pos=False ):
    
    start = time.time()
    
    combi = combi[ [ 'msno', 'song_id', 'target' ] ]
    combi['value'] = 1.0
    combi['target'] = combi['target'].fillna( 0 )
    if pos:
        combi['value'] = combi['value'] + combi['target']
            
    print( 'created user features in ',(time.time() - start) )
    
    start = time.time()
    
    
    model = BPR(rank=32, n_users=combi.msno.nunique(), n_items = combi.song_id.nunique())
    model.user_key = 'msno'
    model.item_key = 'song_id'
    
    # train the model on a sparse matrix of item/user/confidence weights
    model.train(combi, 20, 1000)
    
    
    Ufv = model.W.get_value()
    Ifv =  model.H.get_value()
    
    UF = ['uf_'+str(i) for i in range(size)]
    SF = ['sf_'+str(i) for i in range(size)]
    
    Uf = pd.DataFrame( Ufv, index=range(combi.msno.nunique()) )
    Uf.columns = UF
    Uf['msno'] = Uf.index
    If = pd.DataFrame( Ifv, index=range(combi.song_id.nunique()) )
    If.columns = SF
    If['song_id'] = If.index
     
    Uf.to_csv(folder+'bpr_user_features'+('_pos' if pos else '')+'.'+str(size)+'.csv',index=False)
    If.to_csv(folder+'bpr_song_features'+('_pos' if pos else '')+'.'+str(size)+'.csv',index=False)
    
    print('created latent social features in ',(time.time() - start))
    
    res = []
          
    for row in combi.itertuples():
        
        song = Ifv[row.song_id]
        user = Ufv[row.msno]
        
        res.append( np.dot( user, song.T ) )
        
    combi['reconst'] = res
    
    print( combi[['msno','song_id','target','value','reconst']] )

def create_latent_factors_artist( combi, folder=FOLDER, size=10, pos=False ):
    
    start = time.time()
    
    combi = combi[ [ 'msno', 'artist_name', 'target' ] ]
    combi['value'] = 1.0
    combi['target'] = combi['target'].fillna( 0 )
    if pos:
        combi['value'] = combi['value'] + combi['target']
            
    print( 'created user features in ',(time.time() - start) )
    
    start = time.time()
     
    model = BPR(rank=32, n_users=combi.msno.nunique(), n_items = combi.artist_name.nunique())
    model.user_key = 'msno'
    model.item_key = 'artist_name'
    
    # train the model on a sparse matrix of item/user/confidence weights
    model.train(combi, 20, 1000)
    
    Ufv = model.W.get_value()
    Afv =  model.H.get_value()
    
    UF = ['uf2_'+str(i) for i in range(size)]
    AF = ['af_'+str(i) for i in range(size)]
    
    Uf = pd.DataFrame( Ufv, index=range(combi.msno.nunique()) )
    Uf.columns = UF
    Uf['msno'] = Uf.index
    Af = pd.DataFrame( Afv, index=range(combi.artist_name.nunique()) )
    Af.columns = AF
    Af['artist_name'] = Af.index
     
    Uf.to_csv(folder+'bpr_user2_features'+('_pos' if pos else '')+'.'+str(size)+'.csv',index=False)
    Af.to_csv(folder+'bpr_artist_features'+('_pos' if pos else '')+'.'+str(size)+'.csv',index=False)
    
    print('created latent social features in ',(time.time() - start))
    
    res = []
          
    for row in combi.itertuples():
        
        song = Afv[row.artist_name]
        user = Ufv[row.msno]
        
        res.append( np.dot( user, song.T ) )
        
    combi['reconst'] = res
    
    print( combi[['msno','artist_name','target','value','reconst']] )
  
if __name__ == '__main__':
    
    combi = features.load_combi_prep( FOLDER )
    create_latent_factors( combi, size=32, pos=False )
    create_latent_factors_artist( combi, size=32, pos=False )
    