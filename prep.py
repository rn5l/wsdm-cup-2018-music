from builtins import str
import time

from helper import features

FOLDER = 'data_split001/'
SINGLE = True

if __name__ == '__main__':
    
    start = time.time()
        
    if SINGLE:
        combi = features.load_combi( FOLDER )
        print( combi.info() )
        combi.to_pickle(FOLDER + 'combi_extra.pkl')
    else: 
        for i in range(5):
            combi = features.load_combi( FOLDER, split=i )
            combi.to_pickle(FOLDER + 'combi_extra.'+str(i)+'.pkl')
    
    
    
