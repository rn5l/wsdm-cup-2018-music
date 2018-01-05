#LATENT VECTOR SIZES
UF_SIZE = 32
UF2_SIZE = 32
SF_SIZE = 32
AF_SIZE = 32

#CLUSTER SIZES
UC_SIZE = 25
SC_SIZE = 25
USC_SIZE = 25

ID_FEATURES = ['msno','song_id']

CAT_FEATURES_USER = [
    'bd', 
    'city', 
    'gender', 
    'language', 
    'registered_via',
    ]
CAT_FEATURES_SONG = [
    'artist_name',
    'composer',
    'genre_max',
    'isrc',
    'isrc_country',
    'lyricist',
    'name',
    ]
CAT_FEATURES_GENERAL = [
    'source_screen_name', 
    'source_system_tab',
    'source_type',
    ]
CAT_FEATURES = CAT_FEATURES_USER + CAT_FEATURES_SONG + CAT_FEATURES_GENERAL

BASIC_FEATURES = [
       'expiration_date',
       'expiration_day',
       'expiration_month',
       'expiration_year', 
       'registration_date',  
       'registration_day', 
       'registration_month', 
       'registration_year',
       'song_length',
       'isrc_year',
       ]

BASIC_TIME_FEATURES = [
       'membership_days',
       'membership_diff',
      ]

USER_FEATURES = [
       'u_played', 
       'u_played_rel',
       'u_played_rel_global',
       ]
USER_FEATURES_TIME = [
       'u_time_min',
       'u_time_max',
       'u_time_mean',
       ]
USER_FEATURES_TILL = [
       'u_played_till', 
       'u_played_till_rel',
       ]
USER_FEATURES_SET = [
        'u_played_train',
        'u_played_test',
    ]
USER_FEATURES_TIME5 = [
        'ubin5_played',
        'ubin5_played_rel',
        'ubin5_played_rel_global',
    ]
USER_FEATURES_TIME10 = [
        'ubin10_played',
        'ubin10_played_rel',
        'ubin10_played_rel_global',
    ]
USER_FEATURES_TIME20 = [
        'ubin20_played',
        'ubin20_played_rel',
        'ubin20_played_rel_global',
    ]
USER_FEATURES_TIME30 = [
        'ubin30_played',
        'ubin30_played_rel',
        'ubin30_played_rel_global',
    ]
USER_FEATURES_RATIO = [
       'u_played_train',
       'u_in_train',
       'u_train_ratio',
       ]
USER_FEATURES_POS = [
       'u_played_pos', 
       'u_played_pos_rel', 
       'u_played_ratio',
       ]
USER_ARTIST_FEATURES = [
       'ua_played',
       'ua_played_rel',
       'ua_played_rel2',
       'ua_played_rel_global'
       ]
USER_ARTIST_FEATURES_RATIO = [
       'ua_played_train',
       'ua_in_train',
       'ua_train_ratio',
       ]
USER_ARTIST_FEATURES_POS = [
       'ua_played_pos',
       'ua_played_pos_rel',
       'ua_played_ratio',
       'ua_played_pos_rel_pos',
    ]

USER_GENRE_FEATURES = [
       'ug_played',
       'ug_played_rel',
       'ug_played_rel2',
       'ug_played_rel_global',
       ]
USER_GENRE_FEATURES_RATIO = [
       'ug_played_train',
       'ug_in_train',
       'ug_train_ratio',
       ]
USER_GENRE_FEATURES_POS = [
       'ug_played_pos',
       'ug_played_pos_rel',
       'ug_played_ratio',
       'ug_played_pos_rel_pos'
    ]

USER_SOURCE_TAB_FEATURES = [
       'usst_played',
       'usst_played_rel',
       'usst_played_rel2',
       'usst_played_rel_global',
       ]
USER_SOURCE_TAB_FEATURES_RATIO = [
       'usst_played_train',
       'usst_in_train',
       'usst_train_ratio',
       ]
USER_SOURCE_TAB_FEATURES_POS = [
       'usst_played_pos',
       'usst_played_pos_rel',
       'usst_played_ratio',
       'usst_played_pos_rel_pos'
    ]

USER_SOURCE_TYPE_FEATURES = [
       'ust_played',
       'ust_played_rel',
       'ust_played_rel2',
       'ust_played_rel_global',
       ]
USER_SOURCE_TYPE_FEATURES_RATIO = [
       'ust_played_train',
       'ust_in_train',
       'ust_train_ratio',
       ]
USER_SOURCE_TYPE_FEATURES_POS = [
       'ust_played_pos',
       'ust_played_pos_rel',
       'ust_played_ratio',
       'ust_played_pos_rel_pos'
    ]

USER_SOURCE_NAME_FEATURES = [
       'usn_played',
       'usn_played_rel',
       'usn_played_rel2',
       'usn_played_rel_global',
       ]
USER_SOURCE_NAME_FEATURES_RATIO = [
       'usn_played_train',
       'usn_in_train',
       'usn_train_ratio',
       ]
USER_SOURCE_NAME_FEATURES_POS = [
       'usn_played_pos',
       'usn_played_pos_rel',
       'usn_played_ratio',
       #'usn_played_pos_rel_pos'
    ]
SONG_FEATURES = [
        #'song_id',
        's_played',
        's_played_rel',
        's_played_rel_global',
    ]
SONG_FEATURES_TIME = [
        's_time_min',
        's_time_max',
        's_time_mean',
    ]
SONG_FEATURES_TILL = [
       's_played_till', 
       's_played_till_rel',
       ]
SONG_FEATURES_SET = [
        's_played_train',
        's_played_test',
    ]
SONG_FEATURES_TIME5 = [
        'sbin5_played',
        'sbin5_played_rel',
        'sbin5_played_rel_global',
    ]
SONG_FEATURES_TIME10 = [
        'sbin10_played',
        'sbin10_played_rel',
        'sbin10_played_rel_global',
    ]
SONG_FEATURES_TIME20 = [
        'sbin20_played',
        'sbin20_played_rel',
        'sbin20_played_rel_global',
    ]
SONG_FEATURES_TIME30 = [
        'sbin30_played',
        'sbin30_played_rel',
        'sbin30_played_rel_global',
    ]
SONG_FEATURES_RATIO = [
       's_played_train',
       's_in_train',
       's_train_ratio',
       ]
SONG_FEATURES_POS = [
        's_played_pos', 
        's_played_pos_rel', 
        's_played_ratio',
    ]

SONG_SOURCE_TAB_FEATURES = [
       's_sst_played',
       's_sst_played_rel',
       's_sst_played_rel2',
       's_sst_played_rel_global',
       ]
SONG_SOURCE_TAB_FEATURES_RATIO = [
       's_sst_played_train',
       's_sst_in_train',
       's_sst_train_ratio',
       ]
SONG_SOURCE_TAB_FEATURES_POS = [
       's_sst_played_pos',
       's_sst_played_pos_rel',
       's_sst_played_ratio',
       's_sst_played_pos_rel_pos'
    ]

SONG_SOURCE_TYPE_FEATURES = [
       's_st_played',
       's_st_played_rel',
       's_st_played_rel2',
       's_st_played_rel_global',
       ]
SONG_SOURCE_TYPE_FEATURES_RATIO = [
       's_st_played_train',
       's_st_in_train',
       's_st_train_ratio',
       ]
SONG_SOURCE_TYPE_FEATURES_POS = [
       's_st_played_pos',
       's_st_played_pos_rel',
       's_st_played_ratio',
       's_st_played_pos_rel_pos'
    ]

SONG_SOURCE_NAME_FEATURES = [
       's_sn_played',
       's_sn_played_rel',
       's_sn_played_rel2',
       's_sn_played_rel_global',
       ]
SONG_SOURCE_NAME_FEATURES_RATIO = [
       's_sn_played_train',
       's_sn_in_train',
       's_sn_train_ratio',
       ]
SONG_SOURCE_NAME_FEATURES_POS = [
       's_sn_played_pos',
       's_sn_played_pos_rel',
       's_sn_played_ratio',
       's_sn_played_pos_rel_pos'
    ]

SONG_GENDER_FEATURES = [
       's_gen_played',
       's_gen_played_rel',
       's_gen_played_rel2',
       's_gen_played_rel_global',
       ]
SONG_GENDER_FEATURES_RATIO = [
       's_gen_played_train',
       's_gen_in_train',
       's_gen_train_ratio',
       ]
SONG_GENDER_FEATURES_POS = [
       's_gen_played_pos',
       's_gen_played_pos_rel',
       's_gen_played_ratio',
       's_gen_played_pos_rel_pos'
    ]

ARTIST_FEATURES = [
        'a_played', 
        'a_played_rel',
        'a_played_rel_global',    
    ]
ARTIST_FEATURES_TILL = [
       'a_played_till', 
       'a_played_till_rel',
       ]
ARTIST_FEATURES_RATIO = [
       'a_played_train',
       'a_in_train',
       'a_train_ratio',
       ]
ARTIST_FEATURES_POS = [
        'a_played_pos', 
        'a_played_pos_rel', 
        'a_played_ratio',
    ]

ARTIST_GENDER_FEATURES = [
        'a_gen_played', 
        'a_gen_played_rel',
        'a_gen_played_rel2',
        'a_gen_played_rel_global'   
    ]
ARTIST_GENDER_FEATURES_RATIO = [
       'a_gen_played_train',
       'a_gen_in_train',
       'a_gen_train_ratio',
       ]
ARTIST_GENDER_FEATURES_POS = [
        'a_gen_played_pos', 
        'a_gen_played_pos_rel', 
        'a_gen_played_ratio',
    ]

ARTIST_FEATURES_MAX = [
        'amax_played', 
        'amax_played_rel',    
    ]
ARTIST_FEATURES_MAX_RATIO = [
       'amax_played_train',
       'amax_in_train',
       'amax_train_ratio',
       ]
ARTIST_FEATURES_MAX_POS = [
        'amax_played_pos', 
        'amax_played_pos_rel', 
        'amax_played_ratio',
    ]

ARTIST_FEATURES_FIRST = [
        'af1_played', 
        'af1_played_rel',    
    ]
ARTIST_FEATURES_FIRST_RATIO = [
       'af1_played_train',
       'af1_in_train',
       'af1_train_ratio',
       ]
ARTIST_FEATURES_FIRST_POS = [
        'af1_played_pos', 
        'af1_played_pos_rel', 
        'af1_played_ratio',
    ]

GENRE_FEATURES = [
        'g_played', 
        #'g_played_pos', 
        'g_played_rel',
        #'g_played_rel_global',
        #'g_played_pos_rel', 
        #'g_played_ratio',
        ]
GENRE_FEATURES_MAX = [     
        'gmax_played', 
        'gmax_played_pos', 
        'gmax_played_rel',
        'gmax_played_rel_global',
        'gmax_played_pos_rel', 
        'gmax_played_ratio',
        ]
GENRE_FEATURES_TILL = [
       'gmax_played_till', 
       'gmax_played_till_rel',
       ]
GENRE_FEATURES_FIRST = [    
        'gf1_played', 
        #'gf1_played_pos', 
        'gf1_played_rel',
        #'gf1_played_pos_rel', 
        #'gf1_played_ratio'
        ]

SOURCE_NAME_FEATURES = [
        'sn_played', 
        'sn_played_pos', 
        'sn_played_rel',
        'sn_played_rel_global',
        'sn_played_pos_rel', 
        'sn_played_ratio'
        ]

SOURCE_TAB_FEATURES = [
        'sst_played', 
        'sst_played_pos', 
        'sst_played_rel',
        'sst_played_rel_global',
        'sst_played_pos_rel', 
        'sst_played_ratio'
        ]

SOURCE_TYPE_FEATURES = [
        'st_played', 
        'st_played_pos', 
        'st_played_rel',
        'st_played_pos_rel', 
        'st_played_ratio'
        ]

CITY_FEATURES = [
        'c_played', 
        'c_played_pos', 
        'c_played_rel',
        'c_played_rel_global',
        'c_played_pos_rel', 
        'c_played_ratio'
        ]

COMPOSER_FEATURES = [
        'comp_played', 
        'comp_played_rel',
        'comp_played_rel_global',
        ]
COMPOSER_FEATURES_RATIO = [
       'comp_played_train',
       'comp_in_train',
       'comp_train_ratio',
       ]
COMPOSER_FEATURES_POS = [
        'comp_played_pos', 
        'comp_played_pos_rel', 
        'comp_played_ratio'
        ]

COMPOSER_FEATURES_MAX = [
        'compmax_played', 
        'compmax_played_rel',
        'compmax_played_rel_global',
        ]
COMPOSER_FEATURES_MAX_RATIO = [
       'compmax_played_train',
       'compmax_in_train',
       'compmax_train_ratio',
       ]
COMPOSER_FEATURES_MAX_POS = [
        'compmax_played_pos', 
        'compmax_played_pos_rel', 
        'compmax_played_ratio'
        ]

COMPOSER_FEATURES_FIRST = [
        'compf1_played', 
        'compf1_played_rel',
        'compf1_played_rel_global',
        ]
COMPOSER_FEATURES_FIRST_RATIO = [
       'compf1_played_train',
       'compf1_in_train',
       'compf1_train_ratio',
       ]
COMPOSER_FEATURES_FIRST_POS = [
        'compf1_played_pos', 
        'compf1_played_pos_rel', 
        'compf1_played_ratio'
        ]

LYRICIST_FEATURES = [
        'ly_played', 
        'ly_played_rel',
        'ly_played_rel_global',
        ]
LYRICIST_FEATURES_RATIO = [
       'ly_played_train',
       'ly_in_train',
       'ly_train_ratio',
       ]
LYRICIST_FEATURES_POS = [
        'ly_played_pos', 
        'ly_played_pos_rel', 
        'ly_played_ratio'
        ]

LYRICIST_FEATURES_MAX = [
        'lymax_played', 
        'lymax_played_rel',
        'lymax_played_rel_global',
        ]
LYRICIST_FEATURES_MAX_RATIO = [
       'lymax_played_train',
       'lymax_in_train',
       'lymax_train_ratio',
       ]
LYRICIST_FEATURES_MAX_POS = [
        'lymax_played_pos', 
        'lymax_played_pos_rel', 
        'lymax_played_ratio'
        ]

LYRICIST_FEATURES_FIRST = [
        'lyf1_played', 
        'lyf1_played_rel',
        'lyf1_played_rel_global',
        ]
LYRICIST_FEATURES_FIRST_RATIO = [
       'lyf1_played_train',
       'lyf1_in_train',
       'lyf1_train_ratio',
       ]
LYRICIST_FEATURES_FIRST_POS = [
        'lyf1_played_pos', 
        'lyf1_played_pos_rel', 
        'lyf1_played_ratio'
        ]

AGE_FEATURES = [
        'age_played', 
        'age_played_pos', 
        'age_played_rel',
        'age_played_rel_global',
        'age_played_pos_rel', 
        'age_played_ratio'
        ]

REG_VIA_FEATURES = [
        'rv_played', 
        'rv_played_pos', 
        'rv_played_rel',
        'rv_played_rel_global',
        'rv_played_pos_rel', 
        'rv_played_ratio'
        ]

LANGUAGE_FEATURES = [
        'lang_played', 
        'lang_played_pos', 
        'lang_played_rel',
        'lang_played_rel_global',
        'lang_played_pos_rel', 
        'lang_played_ratio'
        ]

GENDER_FEATURES = [
        'gen_played', 
        'gen_played_pos', 
        'gen_played_rel',
        'gen_played_rel_global',
        'gen_played_pos_rel', 
        'gen_played_ratio'
        ]

UC_FEATURES = [
        'uc_played', 
        'uc_played_pos', 
        'uc_played_rel',
        'uc_played_rel_global',
        'uc_played_pos_rel', 
        'uc_played_ratio'
        ]

UC_ARTIST_FEATURES = [
        'uca_played', 
        #'uca_played_pos', 
        'uca_played_rel',
        'uca_played_rel2',
        'uca_played_rel_global',
        #'uca_played_pos_rel', 
        #'uca_played_ratio'
        ]

UC_GENRE_FEATURES = [
        'ucg_played', 
        #'ucg_played_pos', 
        'ucg_played_rel',
        'ucg_played_rel2',
        'ucg_played_rel_global',
        #'ucg_played_pos_rel', 
        #'ucg_played_ratio'
        ]

SC_FEATURES = [
        'sc_played', 
        'sc_played_pos', 
        'sc_played_rel',
        'sc_played_rel_global',
        #'sc_played_pos_rel', 
        'sc_played_ratio'
        ]

AC_FEATURES = [
        'ac_played', 
        'ac_played_pos', 
        'ac_played_rel',
        'ac_played_rel_global',
        #'ac_played_pos_rel', 
        'ac_played_ratio'
        ]

USC_FEATURES = [
        'usc_played', 
        #'usc_played_pos', 
        'usc_played_rel',
        'usc_played_rel_global',
        #'usc_played_pos_rel', 
        #'usc_played_ratio'
        ]


COS_SIM_FEATURES = [
        's_sim',
        's_sim_pos',
        's_sim_neg',
        'u_sim',
        'u_sim_pos',
        'u_sim_neg'
        ]

SIM_FEATURES = [
        'sim_cosine',
        'sim_cosine_user',
        #'sim_cosine_user_pos',
        'sim_cosine_item',
        #'sim_cosine_item_pos',
        'sim_euclid',
        'sim_manhattan',
        #'sim_score',
        #'sim_size'
        ]

ASIM_FEATURES = [
        'asim_cosine',
        'asim_euclid',
        'asim_manhattan',
        'asim_score',
        'asim_size'
        ]

EXTRA_SPOTIFY = ['acousticness', 'danceability', 'energy', 'instrumentalness',
 'loudness', 'mode', 'popularity', 'speechiness', 'tempo',
 'time_signature', 'valence']

EXTRA_MUSICBRANIZ = ['tags_mb', 'rating', 'votes', 'tags_mb_first', 'tags_mb_max']

LATENT_USER_FEATURES = ['uf_'+str(i) for i in range(UF_SIZE)]
LATENT_USER2_FEATURES = ['uf2_'+str(i) for i in range(UF2_SIZE)]
LATENT_SONG_FEATURES = ['sf_'+str(i) for i in range(SF_SIZE)]
LATENT_ARTIST_FEATURES = ['af_'+str(i) for i in range(AF_SIZE)]

FEATURES = []

'''BASIC FEATURES'''
FEATURES = FEATURES + CAT_FEATURES 
FEATURES = FEATURES + BASIC_FEATURES

'''STATISTICS FEATURES'''
FEATURES = FEATURES + USER_FEATURES #+ USER_FEATURES_RATIO + USER_FEATURES_POS 
FEATURES = FEATURES + USER_ARTIST_FEATURES #+ USER_ARTIST_FEATURES_RATIO #+ USER_ARTIST_FEATURES_POS
FEATURES = FEATURES + USER_GENRE_FEATURES# + USER_GENRE_FEATURES_RATIO #+ USER_GENRE_FEATURES_POS
FEATURES = FEATURES + USER_SOURCE_TAB_FEATURES #+ USER_SOURCE_TAB_FEATURES_RATIO #+ USER_SOURCE_TAB_FEATURES_POS
FEATURES = FEATURES + USER_SOURCE_TYPE_FEATURES #+ USER_SOURCE_TYPE_FEATURES_RATIO #+ USER_SOURCE_TYPE_FEATURES_POS
FEATURES = FEATURES + USER_SOURCE_NAME_FEATURES #+ USER_SOURCE_NAME_FEATURES_RATIO #+ USER_SOURCE_NAME_FEATURES_POS
   
FEATURES = FEATURES + SONG_FEATURES #+ SONG_FEATURES_RATIO #+ SONG_FEATURES_POS
FEATURES = FEATURES + SONG_SOURCE_TAB_FEATURES #+ SONG_SOURCE_TAB_FEATURES_RATIO #+ SONG_SOURCE_TAB_FEATURES_POS
FEATURES = FEATURES + SONG_SOURCE_TYPE_FEATURES #+ SONG_SOURCE_TYPE_FEATURES_RATIO #+ SONG_SOURCE_TYPE_FEATURES_POS
FEATURES = FEATURES + SONG_SOURCE_NAME_FEATURES #+ SONG_SOURCE_NAME_FEATURES_RATIO #+ SONG_SOURCE_NAME_FEATURES_POS
FEATURES = FEATURES + SONG_GENDER_FEATURES #+ SONG_GENDER_FEATURES_RATIO #+ SONG_GENDER_FEATURES_POS
   
FEATURES = FEATURES + ARTIST_FEATURES #+ ARTIST_FEATURES_RATIO #+ ARTIST_FEATURES_POS
#FEATURES = FEATURES + ARTIST_FEATURES_TILL 
#FEATURES = FEATURES + ARTIST_FEATURES_MAX #+ ARTIST_FEATURES_MAX_POS
#FEATURES = FEATURES + ARTIST_FEATURES_FIRST #+ ARTIST_FEATURES_FIRST_POS
#FEATURES = FEATURES + ARTIST_GENDER_FEATURES #+ ARTIST_GENDER_FEATURES_RATIO #+ ARTIST_GENDER_FEATURES_POS
   
#FEATURES = FEATURES + GENRE_FEATURES
FEATURES = FEATURES + GENRE_FEATURES_MAX #pos included
#FEATURES = FEATURES + GENRE_FEATURES_FIRST
   
FEATURES = FEATURES + LANGUAGE_FEATURES #pos included
FEATURES = FEATURES + AGE_FEATURES  #pos included
FEATURES = FEATURES + GENDER_FEATURES  #pos included
#FEATURES = FEATURES + REG_VIA_FEATURES  #pos included
FEATURES = FEATURES + COMPOSER_FEATURES #+ COMPOSER_FEATURES_RATIO + COMPOSER_FEATURES_POS
FEATURES = FEATURES + LYRICIST_FEATURES #+ LYRICIST_FEATURES_RATIO #+ LYRICIST_FEATURES_POS
   
FEATURES = FEATURES + SOURCE_NAME_FEATURES #pos included
FEATURES = FEATURES + SOURCE_TAB_FEATURES #pos included
FEATURES = FEATURES + SOURCE_TYPE_FEATURES #pos included
FEATURES = FEATURES + CITY_FEATURES #pos included

'''CLUSTER FEATURES'''
#FEATURES = FEATURES + SC_FEATURES 
FEATURES = FEATURES + UC_FEATURES
#FEATURES = FEATURES + AC_FEATURES
FEATURES = FEATURES + UC_ARTIST_FEATURES
FEATURES = FEATURES + UC_GENRE_FEATURES
FEATURES = FEATURES + USC_FEATURES

'''LATENT FEATURES'''
FEATURES = FEATURES + LATENT_USER_FEATURES 
FEATURES = FEATURES + LATENT_SONG_FEATURES
FEATURES = FEATURES + LATENT_USER2_FEATURES
#FEATURES = FEATURES + LATENT_ARTIST_FEATURES

'''TIME FEATURES'''
FEATURES = FEATURES + BASIC_TIME_FEATURES
#FEATURES = FEATURES + USER_FEATURES_TIME 
#FEATURES = FEATURES + SONG_FEATURES_TIME 
FEATURES = FEATURES + USER_FEATURES_TIME5
#FEATURES = FEATURES + SONG_FEATURES_TIME5
#FEATURES = FEATURES + USER_FEATURES_TILL 
#FEATURES = FEATURES + GENRE_FEATURES_TILL
#FEATURES = FEATURES + SONG_FEATURES_TILL
#FEATURES = FEATURES + ARTIST_FEATURES_TILL

#LEFT OUT EXTRA DATA
#FEATURES = FEATURES + EXTRA_SPOTIFY
#FEATURES = FEATURES + EXTRA_MUSICBRANIZ
#FEATURES = FEATURES + COS_SIM_FEATURES

FEATURES = FEATURES

NECESSARY = ['id','train','target'] + ['msno','song_id']
