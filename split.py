from builtins import str
import math
import time

import pandas as pd


FOLDER = '../kkbox-py/data/'
FOLDER_SPLIT = 'data_split001/'
SINGLE = True
SAMPLE = 0.01

if __name__ == '__main__':
    
    start = time.time()
     
    train = pd.read_csv( FOLDER + 'train.csv')
     
    print( 'loaded in: ', (time.time() - start) )
     
    if SINGLE:
        
        split = math.ceil( len(train) * 0.75 )
        
        test = train[split:]
        train = train[:split]
        
        if SAMPLE < 1 :
            test = test[:(math.ceil( len(test)*SAMPLE ))]
            train = train[-(math.ceil( len(train)*SAMPLE )):]
        
        test['id'] = range(len(test))
        test.to_csv( FOLDER_SPLIT+'test_eval.csv', index=False)
         
        del test['target']
        test.to_csv( FOLDER_SPLIT+'test.csv', index=False)
         
        train.to_csv( FOLDER_SPLIT+'train.csv', index=False)
        
        print( 'split in: ', (time.time() - start) )
         
    else:
        
        split_size = math.ceil( len(train) / 2 )
        test_size = math.ceil( split_size * 0.25 )
        
        for i in range(5):
            trains = train[(i*test_size):(i*test_size + (split_size-test_size))].copy()
            tests = train[(i*test_size + (split_size-test_size)):(i*test_size + split_size)].copy()
             
            tests['id'] = range(len(tests))
            tests.to_csv( FOLDER_SPLIT+'test_eval.'+str(i)+'.csv', index=False)
             
            del tests['target']
            tests.to_csv( FOLDER_SPLIT+'test.'+str(i)+'.csv', index=False)
             
            trains.to_csv( FOLDER_SPLIT+'train.'+str(i)+'.csv', index=False)
        
        print( 'split in: ', (time.time() - start) )
        