<h2> Step 1: Test Mesh Transfer Model
* put train.mp4, test.mp4 in data/data_raw
* run script/preprocess.sh for preprocessing
* create data/train, data/test directories
* move data/data_preprocessed/train.mp4, data/data_preprocessed/test.mp4 into data/train, data/test
* train with train.py, and test self reconstruction with script/self_recon.sh

<h3> Step 2: Upgrade the model
* Given model has limitation of moddifying only in-mask area (can not deal with head pose manipulation)
* How can we be free from such limiatation? (From given single image and mesh image, how to make head rotation?)