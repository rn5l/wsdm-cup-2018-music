import time

from helper.feature_list import FEATURES
import helper.features as fl
import lightgbm as lgbm
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn import metrics
import gc

FOLDER = 'data_split001/'
SOLUTION = 'solutions/test.csv.gz'
EVAL = True #evaluate locally or save solution to file

if __name__ == '__main__':


#     featureset = fl.create_featureset( FOLDER, rows=None, save='feature_set_als_32_cluster25' )
    featureset = fl.load_featureset( FOLDER, name='feature_set_als_32_cluster25' )
    
    train = featureset[ featureset.train == 1 ]
    test = featureset[ featureset.train == 0 ]
    del featureset
    gc.collect()
    
    print( FEATURES )
    
    X = train[ FEATURES ]
    y = train[ 'target' ]
    del train
    gc.collect()
    
    X_train, X_valid, y_train, y_valid = ms.train_test_split( X, y, test_size=0.2, shuffle=False )
    del X_train, y_train
    gc.collect()
    
    d_train = lgbm.Dataset( X[FEATURES], label=y, feature_name=FEATURES)#, categorical_feature=CAT_FEATURES )
    d_valid = lgbm.Dataset( X_valid[FEATURES], label=y_valid, feature_name=FEATURES)#, categorical_feature=CAT_FEATURES )
    
    watchlist = [d_valid]

    params = {}
    params['learning_rate'] = 0.1
    params['application'] = 'binary'
    params['max_depth'] = 15
    params['num_leaves'] = 2**8
    params['verbosity'] = 0
    params['metric'] = 'auc'
    
    start = time.time()
    
    evals_result = {}
    model = lgbm.train( params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, early_stopping_rounds=50, evals_result=evals_result, verbose_eval=10 )
    
    print( str( time.time() - start ), ' for training' )
     
    #ax = lgbm.plot_importance(model, max_num_features=100)
    #plt.show()
    
    model.save_model('model_split_best.mdl', num_iteration=model.best_iteration)
    
    X_test = test[ FEATURES ]
    y_test = model.predict(X_test, num_iteration=model.best_iteration)
    
    solution = pd.DataFrame()
    solution['id'] = test['id'].astype(np.int64)
    solution['target'] = y_test 
    solution.sort_values('id', inplace=True)
    
    if not EVAL:
        solution.to_csv(SOLUTION, sep=',', index=False, compression='gzip', float_format = '%.5f')
    else: 
        eval_set = pd.read_csv(FOLDER+'test_eval.csv')
        
        solution['class'] = solution['target'].apply( lambda x: 1 if x > 0.5 else 0 )
        
        fpr, tpr, thresholds = metrics.roc_curve( eval_set['target'], solution['target'], pos_label=1 )
        auc = metrics.auc(fpr, tpr)
        print( 'AUC: ', auc )
        
        acc = metrics.classification.accuracy_score(eval_set['target'], solution['class'])
        print( 'ACC: ', acc )

    