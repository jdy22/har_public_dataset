Code to play around with the sample dataset for single-target HAR available at https://github.com/ermongroup/Wifi_Activity_Recognition.

Steps:
- Download and unzip dataset from link above.
- Run raw_data_processing.py (taken from link above with a few minor edits).
- Run lstm_preprocessing_test1.py followed by lstm_training_test1.py to train an LSTM model for HAR (same model parameters used but different code written to use Pytorch rather than TensorFlow).

Notes:
- Number of training epochs set to a low number (5) - aim at the moment is to familiarise with the dataset and possible NN architectures rather than model performance.
- For test 1 only two activities are considered to speed up the code.


To run using school GPUs:
1. Download VPN and connect:
Follow instructions here
https: // www. imperial. ac. uk/ admin-services/ ict/ self-service/ connect-communicate/ remote-access/ virtual-private-network-vpn/ 

2. To log into the GPU
> ssh <name>@146.169.4.119   # to ssh into the VM
> passed <name>              # to change pw

3. Creating a separate session (so you can leave the GPU running even when you log out)
> screen -S <sessionname>    # to start a new session, don't include "<>"
> Ctrl A+D.                  # to detach from the session
> screen -ls                 # to list the ongoing sessions
> screen -r <sessionname>    # to reattach back to the session
> exit                       # to close the session completely
