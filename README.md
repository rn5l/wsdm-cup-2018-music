# wsdm-cup-2018-music
Publication of the code we used in the WSDM'18 Cup Music Recommendation Challenge.

## Usage
* Place data files in the data/ and data_common/ directory
* Run the python scripts in the following order
** split.py (Optional, creates a local sample)
** prep_songs.py (Some extra data preparation and combination, requires the cleaning of songs.csv as songs_clean.csv)
** prep.py (Some dataset preparations)
** als.py (Creates the latent features)
** cluster.py (Clusters the latent features for users, tracks, artists, and combinations of those)
** create_featureset.py (Combines all data in one dataset to work on)
** lgbm_test.py (Tests a single LGBM model)
** lgbm_cv.py (Creates the n-fold trained ensemble)
* The folder to do the work in is defined as FOLDER in each script along with other important parameters
* The file helper/feature_list.py defines a list of features that is finally used for model training