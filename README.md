# Introduction:
This repository contains the model training and hyperparameter tuning codes for the LSTM and ABLSTM on the public dataset.

## Datasets:
In order to train or run any of the files in this model repository, the corresponding data files must be located in your directory. 

### Public Dataset:
The public dataset has been provided by Yousefi et al. [1] and can be obtained online via https://github.com/ermongroup/Wifi_Activity_Recognition. 

## Raw Data Preprocessing:
Prior to running any of the individual files in this repository, the raw data preprocessing files located in “collected_data_preprocessing” repository (https://gitlab.doc.ic.ac.uk/g22mai03/collected_data_preprocessing) must be run on the public dataset. Please ensure that any data paths defined in the python files are changed to be compatible with your local directory. A more detailed description on running these files can be found in this repository’s README file. 

### Public Raw Data Preprocessing:
To run the raw data preprocessing on the public dataset, run the following python files:
* Public Data: “raw_data_processing.py” followed by "public_dataset_preprocessing.py"

## Model Training:

### LSTM:
The "lstm_training_test1.py" file will train the LSTM model on the public dataset. The best hyperparameter settings have been saved in this file. The test loss and accuracy per epoch, as well as the final accuracy and confusion matrix will be printed to the terminal. If "logging" in line 106 is set to "True", this information will also be saved to the text file specified in line 107.

### ABLSTM:

## Hyperparameter Tuning:

### LSTM:

The "lstm_grid_search.py" file will run a grid search on the hyperparameter settings for the LSTM model on the public dataset. Results will be printed to the terminal. If "logging" in line 22 is set to "True", this information will also be saved to the text file specified in line 23.

### ABLSTM:

## Archives:
Old code files and logs have been stored in the archives folder of this repository for record-purposes.

## References:
[1] Siamak Yousefi, Hirokazu Narui, Sankalp Dayal, Stefano Ermon, and Shahrokh Valaee. A survey on behavior recognition using wifi channel state information. IEEE Communications Magazine, 55(10):98–104, 2017.
