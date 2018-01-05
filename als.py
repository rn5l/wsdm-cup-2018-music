from helper import features
from mf import implicit_features as mf

FOLDER = 'data_split001/'

if __name__ == '__main__':
        
    combi = features.load_combi_prep( FOLDER )
    mf.create_latent_factors( combi, folder=FOLDER, size=32, pos=False )
    mf.create_latent_factors_artist( combi, folder=FOLDER, size=32, pos=False )