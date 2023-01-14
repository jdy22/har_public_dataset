Code to play around with the sample dataset for single-target HAR available at https://github.com/ermongroup/Wifi_Activity_Recognition.

Steps:
- Download and unzip dataset from link above.
- Run raw_data_processing.py (taken from link above with a few minor edits).
- Run lstm_preprocessing_test1.py followed by lstm_training_test1.py to train an LSTM model for HAR (same model parameters used but different code written to use Pytorch rather than TensorFlow).

Notes:
- Number of training epochs set to a low number (5) - aim at the moment is to familiarise with the dataset and possible NN architectures rather than model performance.
- For test 1 only two activities are considered to speed up the code.