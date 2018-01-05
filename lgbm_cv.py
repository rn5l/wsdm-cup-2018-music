import gc
import time

from helper.feature_list import FEATURES
import helper.features as fl
import lightgbm as lgbm
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import sklearn.model_selection as ms


FOLDER = 'data_split001/'
SOLUTION = 'solutions/test_cv_10_shuffle.csv'
EVAL = True #evaluate locally or save solution to file

if __name__ == '__main__':

    start = time.time()
    
#     featureset = fl.create_featureset( FOLDER, rows=None, save='feature_set' )
    featureset = fl.load_featureset( FOLDER, name='feature_set_als_32_cluster25' )
    
    train = featureset[ featureset.train == 1 ]
    test = featureset[ featureset.train == 0 ]
    del featureset
    gc.collect()
    
    X = train[ FEATURES ].values
    X_test = test[ FEATURES ].values
    
    y = train[ 'target' ].values
    ids = test['id'].astype( np.int64 )
    
    del train, test
    gc.collect()
    
    folds = 10
    eval_part = 0.2
    
    cv = ms.KFold( n_splits=folds, shuffle=True )
    X_train, X_valid, y_train, y_valid = ms.train_test_split( X, y, test_size=eval_part, shuffle=False )
    del X_train, y_train
    gc.collect()
    
    res = 0
    res2 = 0
    
    predictions = np.array( [ 0.0 for i in range( len( ids ) ) ] )
    
    #del test,train
    
    for train_idx, test_idx in cv.split(X, y): 
                
        d_train = lgbm.Dataset( X[train_idx], y[train_idx], feature_name=FEATURES)#, categorical_feature=CAT_FEATURES )
        d_valid = lgbm.Dataset( X_valid, y_valid, feature_name=FEATURES)#, categorical_feature=CAT_FEATURES )
        
        watchlist = [d_train, d_valid]
    
        params = {}
        params['learning_rate'] = 0.1
        params['application'] = 'binary'
        params['max_depth'] = 15
        params['num_leaves'] = 2**8
        params['verbosity'] = 0
        params['metric'] = 'auc'
        
    #     res = lgbm.cv( params, d_train, 200, nfold=5, verbose_eval=50 )
    #     
    #     ax = lgbm.plot_metric(res, metric='auc')
    #     plt.show()
        
        evals_result = {}
        model = lgbm.train( params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, early_stopping_rounds=50, evals_result=evals_result, verbose_eval=10 )
        
        y_test = model.predict(X_test, num_iteration=model.best_iteration)
        predictions += y_test
        
        del evals_result
        
        gc.collect()
        
   
    print( FEATURES )
    
    predictions = predictions / folds
        
    solution = pd.DataFrame()
    solution['id'] = ids
    solution['target'] = predictions
            
    if not EVAL:
        solution.to_csv(SOLUTION, sep=',', index=False, compression='gzip', float_format = '%.6f')
    else: 
        eval_set = pd.read_csv(FOLDER+'test_eval.csv')
        
        solution['class'] = solution['target'].apply( lambda x: 1 if x > 0.5 else 0 )
        
        fpr, tpr, thresholds = metrics.roc_curve( eval_set['target'], solution['target'], pos_label=1 )
        auc = metrics.auc(fpr, tpr)
        print( 'AUC: ', auc )
        
        acc = metrics.classification.accuracy_score(eval_set['target'], solution['class'])
        print( 'ACC: ', acc )
        