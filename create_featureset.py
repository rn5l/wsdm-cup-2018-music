import helper.features as fs

FOLDER = 'data_split001/'

if __name__ == '__main__':
    fs.create_featureset(FOLDER, save='feature_set_als_32_cluster25')