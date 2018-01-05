from helper import features
from cluster import cluster_emb as cluster

FOLDER = 'data_split001/'

if __name__ == '__main__':
        
    combi = features.load_combi_prep(FOLDER)
    
    cluster.create_cluster_overlap(combi, folder=FOLDER, size=25, esize=32, prefix='als')
    cluster.create_cluster_song_overlap(combi, folder=FOLDER, size=25, esize=32, prefix='als')
    cluster.create_cluster_artist_overlap(combi, folder=FOLDER, size=25, esize=32, prefix='als')
    cluster.create_cluster_both_overlap(combi, folder=FOLDER, size=25, esize=32, prefix='als')