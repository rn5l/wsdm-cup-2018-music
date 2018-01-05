import pandas as pd
import numpy as np
import time
from helper.feature_list import UF_SIZE, SF_SIZE, UC_SIZE, SC_SIZE, FEATURES, NECESSARY, USC_SIZE, UF2_SIZE, AF_SIZE
import gc

FOLDER_COMMON = 'data_common/'

def load_combi( folder = 'data/', sample=1, rows=None, split=None ):
    
    print( 'LOAD COMBI' )
    
    start = time.time()
    
    if split is None:
        ntest = 'test.csv'
        ntrain = 'train.csv'
    else:
        ntest = 'test.'+str(split)+'.csv'
        ntrain = 'train.'+str(split)+'.csv'
    
    if rows is None:
        test = pd.read_csv( folder + ntest )
        train = pd.read_csv( folder + ntrain )
    else: 
        test = pd.read_csv( folder + ntest, nrows=rows )
        train = pd.read_csv( folder + ntrain, nrows=rows )
    
    if sample < 1:
        train = train[-(len(train) * sample):]
        test = test[:(len(test) * sample)]
    
    print( 'loaded in: ', (time.time() - start) )
    
    train['time'] = train.index
    test['time'] = train['time'].max() + 1 + test.index
    
    start = time.time()

    members = pd.read_csv( FOLDER_COMMON + 'members.csv')
    members['registration_date'] = members['registration_init_time']
    del members['registration_init_time']
    members['registration_year'] = members['registration_date'].apply(lambda x: int(str(x)[0:4]))
    members['registration_month'] = members['registration_date'].apply(lambda x: int(str(x)[4:6]))
    members['registration_day'] = members['registration_date'].apply(lambda x: int(str(x)[6:8]))
    members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
    members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
    members['expiration_day'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
    
    members['membership_daysA'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d')
    members['membership_daysB'] = pd.to_datetime(members['registration_date'], format='%Y%m%d')                                          
    members['membership_days'] = (members['membership_daysA'] - members['membership_daysB']).dt.days.astype(int)
    del members['membership_daysA'],  members['membership_daysB']
    
    members['registration_date'] = members.registration_date.astype(np.int32)
    members['expiration_date'] = members.expiration_date.astype(np.int32)
    members['membership_diff'] = members['expiration_date'] - members['registration_date']
    
    songs_new = pd.read_csv( FOLDER_COMMON + 'songs_complete.csv')

    train = train.merge( members, on='msno', how='inner' )
    train = train.merge( songs_new, on='song_id', how='inner' )
    
    test = test.merge( members, on='msno', how='inner' )
    test = test.merge( songs_new, on='song_id', how='inner' )
    
    del songs_new
    
    train['train'] = 1
    test['train'] = 0
    
    print( 'train size:', len(train) )
    print( 'test size:', len(test) )
    
    combi = pd.concat( [train,test] )
    del test
    del train
        
    print( 'merged in: ', (time.time() - start) )
    
    combi['id'] = combi['id'].astype( np.float32 )
    combi['isrc'] = combi['isrc'].astype( 'category' ).cat.codes
    combi['isrc_country'] = combi['isrc_country'].astype( 'category' ).cat.codes
    combi['isrc_year'] = combi['isrc_year'].astype( 'category' ).astype(np.float16)
    combi['msno'] = combi['msno'].astype( 'category' ).cat.codes
    combi['song_id_new'] = combi['song_id_new'].astype( 'category' ).cat.codes
    combi['song_length'] = combi['song_length'].fillna(0).astype( np.int32 )
    combi['name'] = combi['name'].astype( 'category' ).cat.codes
    combi['language'] = combi['language'].astype( 'category' ).cat.codes
    combi['gender'] = combi['gender'].astype( 'category' ).cat.codes
    combi['artist_name'] = combi['artist_name'].astype( 'category' ).cat.codes
    combi['artist_name_first'] = combi['artist_name_first'].astype( 'category' ).cat.codes
    combi['artist_name_max'] = combi['artist_name_max'].astype( 'category' ).cat.codes
    combi['composer'] = combi['composer'].astype( 'category' ).cat.codes
    combi['composer_first'] = combi['composer_first'].astype( 'category' ).cat.codes
    combi['composer_max'] = combi['composer_max'].astype( 'category' ).cat.codes
    combi['lyricist'] = combi['lyricist'].astype( 'category' ).cat.codes
    combi['lyricist_first'] = combi['lyricist_first'].astype( 'category' ).cat.codes
    combi['lyricist_max'] = combi['lyricist_max'].astype( 'category' ).cat.codes
    combi['city'] = combi['city'].astype( 'category' ).cat.codes
    combi['bd'] = combi['bd'].astype( 'category' ).cat.codes
    combi['genre_ids'] = combi['genre_ids'].astype( 'category' ).cat.codes
    combi['genre_first'] = combi['genre_first'].astype( 'category' ).cat.codes
    combi['genre_max'] = combi['genre_max'].astype( 'category' ).cat.codes
    combi['source_screen_name'] = combi['source_screen_name'].astype( 'category' ).cat.codes
    combi['source_system_tab'] = combi['source_system_tab'].astype( 'category' ).cat.codes
    combi['source_type'] = combi['source_type'].astype( 'category' ).cat.codes
    
    combi['registration_year'] = combi['registration_year'].astype( np.int16 )
    combi['registration_month'] = combi['registration_month'].astype( np.int8 )
    combi['registration_day'] = combi['registration_day'].astype( np.int8 )
    
    combi['expiration_day'] = combi['expiration_day'].astype( np.int8 )
    combi['expiration_month'] = combi['expiration_month'].astype( np.int8 )
    combi['expiration_year'] = combi['expiration_year'].astype( np.int16 )
    
    combi['membership_days'] = combi['membership_days'].astype( np.int16 )
    
    #external data left out
    #combi['tags_mb'] = combi['tags_mb'].astype( 'category' ).cat.codes
    #combi['tags_mb_first'] = combi['tags_mb_first'].astype( 'category' ).cat.codes
    #combi['tags_mb_max'] = combi['tags_mb_max'].astype( 'category' ).cat.codes
    
    del combi['song_id']
    combi['song_id'] = combi['song_id_new']
    del combi['song_id_new']
    
    for i, dte in enumerate(combi.dtypes):
        if dte == 'float64':
            combi[ combi.columns[i] ] = combi[ combi.columns[i] ].astype(np.float32)
    
    print( 'converted in: ', (time.time() - start) )
        
    combi.sort_values('time', inplace=True)
        
    return combi

def load_combi_prep( folder = 'data_new/', split=None ):
    
    print( 'LOAD COMBI' )
    
    start = time.time()
    
    name = 'combi_extra' + ( '.' + str(split) if split is not None else '' ) + '.pkl'
    combi = pd.read_pickle( folder + name)
    
    print( 'loaded in: ', (time.time() - start) )
    
    return combi

def create_featureset( folder = 'data/', sample=1, rows=None, split=None, save=None ):
    
    combi = load_combi_prep( folder=folder, split=None )
    
    #partition data by time
    combi['time_bin10'] = pd.cut( combi['time'], 10, labels=range(10) )
    combi['time_bin5'] = pd.cut( combi['time'], 5, labels=range(5) )
    
    #add latent features
    combi = add_als(folder, combi, UF_SIZE, user=True, postfix='', artist=False)
    combi = add_als(folder, combi, SF_SIZE, user=False, postfix='', artist=False)
    combi = add_als(folder, combi, UF2_SIZE, user=True, postfix='', artist=True)
    combi = add_als(folder, combi, AF_SIZE, user=False, postfix='', artist=True)
    
    #add cluster ids
    combi = add_cluster( folder, combi, col='msno', size=UC_SIZE, overlap=True, positive=True, content=False )
    combi = add_cluster( folder, combi, col='song_id', size=SC_SIZE, overlap=True, positive=True, content=False )
    combi = add_cluster( folder, combi, col='artist_name', size=SC_SIZE, overlap=True, positive=True, content=False )
    combi = add_cluster( folder, combi, col='both', size=USC_SIZE, overlap=True, positive=True, content=False )
    
    # USER FEATURES
    combi = scol_features( combi, 'msno', 'u_' )
        
    # SONG FEATURES
    combi = scol_features( combi, 'song_id', 's_' )

    # ARTIST FEATURES
    combi = scol_features( combi, 'artist_name', 'a_' )
    #combi = scol_features( combi, 'artist_name_max', 'amax_' )
    #combi = scol_features( combi, 'artist_name_first', 'af1_' )
    
    # GENRE FEATURES
    #combi = scol_features( combi, 'genre_ids', 'g_' )
    combi = scol_features( combi, 'genre_max', 'gmax_' )
    #combi = scol_features( combi, 'genre_first', 'gf1_' )
    
    # CITY FEATURES
    combi = scol_features( combi, 'city', 'c_' )
    
    # AGE FEATURES
    combi = scol_features( combi, 'bd', 'age_' )
    
    # LANGUAGE FEATURES
    combi = scol_features( combi, 'language', 'lang_' )
    
    # GENDER FEATURES
    combi = scol_features( combi, 'gender', 'gen_' )
    
    # COMPOSER FEATURES
    combi = scol_features( combi, 'composer', 'comp_' )
    #combi = scol_features( combi, 'composer_max', 'compmax_' )
    #combi = scol_features( combi, 'composer_first', 'compf1_' )
                           
    # LYRICIST FEATURES
    combi = scol_features( combi, 'lyricist', 'ly_' )
    #combi = scol_features( combi, 'lyricist_max', 'lymax_' )
    #combi = scol_features( combi, 'lyricist_first', 'lyf1_' )
                           
    # SOURCE NAME FEATURES
    combi = scol_features( combi, 'source_screen_name', 'sn_' )
    
    # SOURCE TAB FEATURES
    combi = scol_features( combi, 'source_system_tab', 'sst_' )
    
    # SOURCE TYPE FEATURES
    combi = scol_features( combi, 'source_type', 'st_' )
    
    # SOURCE TYPE FEATURES
    combi = scol_features( combi, 'registered_via', 'rv_' )
    
    # USER TIME  FEATURES
    #combi = scol_timefeatures(combi, 'msno', 'u_') #tested after competition
    combi = mcol_timefeatures(combi, 'msno', 'ubin5_', rel_prefix='u_', bin='time_bin5' )
    combi = mcol_timefeatures(combi, 'msno', 'ubin10_', rel_prefix='u_', bin='time_bin10' )
    #combi = scol_timefeatures(combi, 'msno', 'ubin20_', rel_prefix='u_', bin='time_bin20' )
    #combi = scol_timefeatures(combi, 'msno', 'ubin30_', rel_prefix='u_', bin='time_bin30' )
    
    # USER ARTIST  FEATURES
    combi = mcol_features(combi, ['msno', 'artist_name'], 'ua_', rel_prefix2='a_' )
    
    # USER GENRE FEATURES
    combi = mcol_features(combi, ['msno', 'genre_max'], 'ug_', rel_prefix2='gmax_' )
    
    # USER SOURCE TAB FEATURES
    combi = mcol_features(combi, ['msno', 'source_system_tab'], 'usst_', rel_prefix2='sst_' )
    
    #USER SOURCE TYPE FEATURES
    combi = mcol_features(combi, ['msno', 'source_type'], 'ust_', rel_prefix2='st_' )
    
    #USER SOURCE NAME FEATURES
    combi = mcol_features(combi, ['msno', 'source_screen_name'], 'usn_', rel_prefix2='sn_' )
    
    #SONG TIME FEATURES
    #combi = scol_timefeatures(combi, 'song_id', 's_') #tested later
    combi = mcol_timefeatures(combi, 'song_id', 'sbin5_', rel_prefix='s_', bin='time_bin5' )
    combi = mcol_timefeatures(combi, 'song_id', 'sbin10_', rel_prefix='s_', bin='time_bin10' )
    #combi = scol_timefeatures(combi, 'song_id', 'sbin20_', rel_prefix='s_', bin='time_bin20' )
    #combi = scol_timefeatures(combi, 'song_id', 'sbin30_', rel_prefix='s_', bin='time_bin30' )
    
    # SONG SOURCE TAB FEATURES
    combi = mcol_features(combi, ['song_id', 'source_system_tab'], 's_sst_', rel_prefix='s_', rel_prefix2='sst_' )
    
    #SONG SOURCE TYPE FEATURES
    combi = mcol_features(combi, ['song_id', 'source_type'], 's_st_', rel_prefix='s_', rel_prefix2='st_' )
    
    #SONG SOURCE NAME FEATURES
    combi = mcol_features(combi, ['song_id', 'source_screen_name'], 's_sn_', rel_prefix='s_', rel_prefix2='sn_' )
    
    #SONG GENDER FEATURES
    combi = mcol_features(combi, ['song_id', 'gender'], 's_gen_', rel_prefix='s_', rel_prefix2='gen_' )
    
    #SONG GENDER FEATURES
    combi = mcol_features(combi, ['artist_name', 'gender'], 'a_gen_', rel_prefix='a_', rel_prefix2='gen_' )
    
    # USER CLUSTER FEATURES
    combi = scol_features( combi, 'cluster_msno_'+str(UC_SIZE), 'uc_' )
    
    # USER CLUSTER ARTIST  FEATURES
    combi = mcol_features(combi, ['cluster_msno_'+str(UC_SIZE), 'artist_name'], 'uca_', rel_prefix='uc_', rel_prefix2='a_'  )
    
    # USER CLUSTER GENRE FEATURES
    combi = mcol_features(combi, ['cluster_msno_'+str(UC_SIZE), 'genre_ids'], 'ucg_', rel_prefix='uc_', rel_prefix2='gmax_'  )
    
    # SONG CLUSTER FEATURES
    combi = scol_features( combi, 'cluster_song_id_'+str(SC_SIZE), 'sc_' )
    
    # USER SONG CLUSTER FEATURES
    combi = scol_features( combi, 'cluster_both_'+str(UC_SIZE), 'usc_' )
    
    # ARTIST CLUSTER FEATURES
    combi = scol_features( combi, 'cluster_artist_name_'+str(UC_SIZE), 'ac_' )
    
    #keep sorted
    combi.sort_values( 'time',inplace=True )
    
    #combi = add_counts( combi ) #tested after competition
    
    if save != None:
        name = save + ( '.'+str(split) if split is not None else '')
        combi.to_pickle( folder + name + '.pkl' )
    
    return combi[ FEATURES + NECESSARY ]

def load_featureset(folder='data/',name='feature_set',split=None):
    
    name = name + ( '.'+str(split) if split is not None else '')
    combi = pd.read_pickle( folder + name + '.pkl' )
    
    return combi[ FEATURES + NECESSARY ]

def scol_features( combi, col, prefix ):
    
    start = time.time()
    
    tmp = pd.DataFrame()
    group = combi.groupby( [col] )
    #group_train = combi[ combi.train==1 ].groupby( [col] )
    group_pos = combi[ combi.target == 1 ].groupby( col )
    tmp[prefix+'played'] = group.size().astype(np.int32)
    #tmp[prefix+'played_train'] = group_train.size().astype(np.int32) #discarded
    #tmp[prefix+'played_test'] = tmp[prefix+'played'] - tmp[prefix+'played_train'] #discarded
    #tmp[prefix+'in_train'] = (tmp[prefix+'played_train'] > 0) #discarded
    #tmp[prefix+'train_ratio'] = tmp[prefix+'played_train'] / tmp[prefix+'played'] #discarded
    tmp[prefix+'played_pos'] = group_pos.size().astype(np.int32)
    tmp[prefix+'played_pos'] = tmp[prefix+'played_pos'].fillna(0)
    tmp[prefix+'played_rel'] = ( tmp[prefix+'played'] / tmp[prefix+'played'].max() ).astype(np.float32)
    tmp[prefix+'played_rel_global'] = ( tmp[prefix+'played'] / len(combi) ).astype(np.float32)
    tmp[prefix+'played_pos_rel'] = ( tmp[prefix+'played_pos'] / tmp[prefix+'played_pos'].max() ).astype(np.float32)
    tmp[prefix+'played_ratio'] = ( tmp[prefix+'played_pos'] / tmp[prefix+'played'] ).astype(np.float32)
    tmp[col] = tmp.index
    
    combi = combi.merge( tmp, how='inner', on=col )
    
    del tmp, group, group_pos
    
    print( col, ' features in: ', (time.time() - start) )
    
    return combi
    
def mcol_features( combi, col, prefix, rel_prefix='u_', rel_prefix2='s_' ):
    
    start = time.time()
    
    tmp = pd.DataFrame()

    group = combi.groupby( col )
    group_pos = combi[combi.target == 1].groupby( col )
    #group_train = combi[combi.train == 1].groupby( col )
    
    for c in col:
        tmp[c] = group[c].min()
        
    tmp[prefix+'played'] = group.size().astype(np.int32)
    #tmp[prefix+'played_pos'] = group_pos.size().astype(np.int32) #discarded
    #tmp[prefix+'played_pos'] = tmp[prefix+'played_pos'].fillna(0) #discarded
    #tmp[prefix+'played_train'] = group_train.size().astype(np.int32) #discarded
    #tmp[prefix+'in_train'] = (tmp[prefix+'played_train'] > 0) #discarded
    #tmp[prefix+'train_ratio'] = tmp[prefix+'played_train'] / tmp[prefix+'played'] #discarded
    #tmp[prefix+'played_ratio'] = ( tmp[prefix+'played_pos'] / tmp[prefix+'played'] ).astype(np.float32) #discarded
    
    combi = combi.merge( tmp, how='inner', on=col )
    combi[prefix+'played_rel'] = ( combi[prefix+'played'] / combi[rel_prefix+'played'] ).astype(np.float32)
    combi[prefix+'played_rel2'] = ( combi[prefix+'played'] / combi[rel_prefix2+'played'] ).astype(np.float32)
    combi[prefix+'played_rel_global'] = ( combi[prefix+'played'] / len(combi) ).astype(np.float32)
    #combi[prefix+'played_pos_rel'] = ( combi[prefix+'played_pos'] / combi[rel_prefix+'played'] ).astype(np.float32) #discarded
    #combi[prefix+'played_pos_rel_pos'] = ( combi[prefix+'played_pos'] / combi[rel_prefix+'played_pos'] ).astype(np.float32) #discarded
    
    del tmp, group, group_pos
    gc.collect()
    
    print( (','.join(col)), ' features in: ', (time.time() - start) )

    return combi

def mcol_timefeatures( combi, col, prefix, rel_prefix='u_', bin='time_bin10' ):
    
    start = time.time()
    
    tmp = pd.DataFrame()
    group = combi.groupby( [col,bin] )
    
    tmp[col] = group[col].min()
    tmp[bin] = group[bin].min()
    
    tmp[prefix+'played'] = group.size().astype(np.int32)
    tmp[prefix+'played_rel'] = ( tmp[prefix+'played'] / tmp[prefix+'played'].max() ).astype(np.float32)
    
    combi = combi.merge( tmp, how='inner', on=[col,bin] )
    combi[prefix+'played_rel_global'] = ( combi[prefix+'played'] / combi[rel_prefix+'played'] ).astype(np.float32)
    
    del tmp, group
    gc.collect()
    
    print( col, ' mcol time features in: ', (time.time() - start) )
    
    return combi

def scol_timefeatures( combi, col, prefix ):
    
    start = time.time()
    
    tmp = pd.DataFrame()
    group = combi.groupby( [col] )
    tmp[prefix+'time_min'] = group['time'].min()
    tmp[prefix+'time_max'] = group['time'].max()
    tmp[prefix+'time_mean'] = group['time'].mean()
    tmp[col] = tmp.index
    combi = combi.merge( tmp, how='inner', on=[col] )
    
    print( col, ' time features in: ', (time.time() - start) )
    
    return combi

def add_cluster( folder, combi, col, size, overlap=True, positive=True, content=False ):
    
    start = time.time()
    name = 'cluster_' + col
    file_name  = 'alsclusterEMB32_' + col
    if content:
        file_name = 'content_'+file_name
    if overlap:
        file_name += '_ol'
    if not positive:
        file_name += '_nopos'
    
    #cluster = pd.read_csv( folder + 'content_' + name +'.{}.csv'.format(size) )
    cluster = pd.read_csv( folder + file_name +'.{}.csv'.format(size) )
    
    cluster[name + '_' + str(size)] = cluster.cluster_id
    del cluster['cluster_id']
    
    print('cluster ', col)
    print(len(combi))
    
    if col is 'both':
        combi = combi.merge( cluster, how='inner', on=['msno','song_id'] )
    else:
        combi = combi.merge( cluster, how='inner', on=col )
    del cluster
    gc.collect()
    
    print(len(combi))
    
    print( 'cluster num features in: ', (time.time() - start) )
    
    return combi

def add_als( folder, combi, size, user=True, pos=False, postfix='' , artist=False):
    
    start = time.time()
    
    if user:
        if artist:
            name = 'user2'
            key = 'uf2_'
        else:
            name = 'user'
            key = 'uf_'
    else:
        if artist:
            name = 'artist'
            key = 'af_'
        else:
            name = 'song'
            key = 'sf_'
    
    features = pd.read_csv( folder + 'als'+postfix+'_' + name +'_features' + ('_pos' if pos else '') + '.{}.csv'.format(size) )
    
    for i in range(size):
        features[ key+str(i) ] = features[ key+str(i) ].astype( np.float32 )
    
    oncol = 'msno' if user else 'song_id' if not artist else 'artist_name'
    combi = combi.merge( features, how='inner', on=oncol )
    
    del features
    gc.collect()
    
    print( 'nmf features added in: ', (time.time() - start) )
    
    return combi


def add_bpr( folder, combi, size, user=True, pos=False, postfix='' , artist=False):
    
    start = time.time()
    
    if user:
        if artist:
            name = 'user2'
            key = 'uf2_'
        else:
            name = 'user'
            key = 'uf_'
    else:
        if artist:
            name = 'artist'
            key = 'af_'
        else:
            name = 'song'
            key = 'sf_'
    
    features = pd.read_csv( folder + 'bpr'+postfix+'_' + name +'_features' + ('_pos' if pos else '') + '.{}.csv'.format(size) )
    
    for i in range(size):
        features[ key+str(i) ] = features[ key+str(i) ].astype( np.float32 )
    
    oncol = 'msno' if user else 'song_id' if not artist else 'artist_name'
    combi = combi.merge( features, how='inner', on=oncol )
    
    del features
    gc.collect()
    
    print( 'bpr features added in: ', (time.time() - start) )
    
    return combi

def add_counts( combi ):

    print('PREP FEATURES')
    start = time.time()
    
    keys = ['msno','song_id','artist_name','genre_max']
    target_col = ['u_played_till','s_played_till','a_played_till','gmax_played_till']
    rel_col = ['u_played','s_played','a_played','gmax_played']
    
    loc = {}
    c_map = {}
    v_map = {}
    for key in keys:
        loc[key] = combi.columns.get_loc(key)
        c_map[key] = {}
        v_map[key] = []
   
    count = 0
    
    for row in combi.itertuples(index=False):
                
        for key in keys:
            if not row[loc[key]] in c_map[key]:
                c_map[key][row[loc[key]]] = 0
            else:
                c_map[key][row[loc[key]]] += 1
            
            v_map[key].append( c_map[key][row[loc[key]]] )
                
        count+=1
      
        if count % 1000000 is 0: 
            print('processed {} rows in {}s '.format(count, (time.time()-start)))
    
    for i in range(len(keys)): 
        k = keys[i]
        col = target_col[i]
        combi[col] = v_map[k]
        combi[col+'_rel'] = combi[col] / combi[rel_col[i]]
        
    return combi

def split_and_max( combi, column, target, sep='|' ):
    
    def count_max( x ):
        # count number of values (since we can have mutliple values separated by '|')
        if type(x) != str:
            return 1
        else:
            return 1 + x.count(sep)
    
    def split(x, n):
        if type(x) != str:
            if n == 1:
                if not np.isnan(x):
                    return int(x)
                else:
                    return x
        else:
            if x.count(sep) >= n-1:
                return x.split(sep)[n-1].strip()
    
    start = time.time()
    
    combi['number_of_values'] = combi[column].apply(count_max)
    
    max_values = combi['number_of_values'].max()
    
    print( 'max number of ',column,': ',max_values )
    print( combi[ combi['number_of_values'] == max_values ][column].values )
        
    columns_val = []
    
    for i in range(1,max_values+1):
        sp_g = lambda x: split(x, i)
        combi['val_'+str(i)] = combi[column].apply(sp_g)
        columns_val.append( 'val_'+str(i) )
        
    longlist = combi[columns_val].values.ravel()
        
    genresf = pd.DataFrame( longlist, columns=['value'] )
    tmp = pd.DataFrame()
    tmp['pop'] = genresf.groupby('value').size() 
    
    columns_pop = []
    
    for i in range(1,max_values+1):
        tmp[ 'val_'+str(i) ] = tmp.index
        combi = combi.merge( tmp, on='val_'+str(i), how='left' )
        combi['valpop_'+str(i)] = combi['pop']
        del combi['pop']
        del tmp[ 'val_'+str(i) ]
        columns_pop.append( 'valpop_'+str(i) )
    
    combi['max_val_pop'] = combi[columns_pop].idxmax(axis=1)
    combi['max_val_pop'] = combi['max_val_pop'].fillna( 'valpop_1' )
    
    i = pd.Series( 'val_'+(combi['max_val_pop'].str.replace( 'valpop_','' )) )
    combi[target] = combi.lookup( i.index, i.values )  
    
    for i in range(1,max_values+1):
        del combi[ 'val_'+str(i) ]
        del combi[ 'valpop_'+str(i) ]
        
    del combi['max_val_pop']
    del combi['number_of_values']
    
    print( 'converted ',column,' to ',target,' in: ', (time.time() - start) )
    
    return combi

def split_and_first( combi, column, target, sep='|' ):

    def split_or_first(x):
        if type(x) != str:
            if not np.isnan(x):
                return int(x)
            else:
                return x
        else:
            if x.count(sep) >= 1:
                return x.split(sep)[0].strip()
            else:
                return x.strip()
    
    start = time.time()
    
    combi[target] = combi[column].apply( split_or_first )
            
    print( 'converted ',column,' to ',target,' in: ', (time.time() - start) )
    
    return combi
