# DeepSuccinylSite

DeepSuccin : Prediction of protein succinylation sites with deep learning model. Devoloped in KC lab.
# Requirement
  Backend = Tensorflow <br/>
  Keras <br/>
  Numpy <br/>
  Biopython <br/>
  Sklearn <br/>
  Imblearn <br/>
 # Dataset
 Dataset is in FASTA format which includes protein window size of 33. Test dataset is provided. There are two dataset for positive and negative examples.
 # Model
 The best model has been included. Download all 4 parts and extracting the 1st part will automatically extract other three. The final file is named model.h5
 # Prediction for given test dataset
 With all the prerequisite installed, run -> model_load.py
 # Prediction for your dataset
 The format should be same as the test dataset which is basically FASTA format. This model works for window size 33 only. 
 # Contact 
 Feel free to contact us if you need any help : dbkc@mtu.edu
