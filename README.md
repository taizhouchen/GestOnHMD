# GestOnHMD
 Dataset and training code for paper: 

### GestOnBoard: Enabling Gesture-based Interaction on Low-cost VR Head-Mounted Display

The repository contains our dataset and code for on-cardboard gesture recognition. 

## Parameters for training: 

- `-data_path` Path to data dictionary. A valid data dictionary hierarchy is: data/[user_ID]/[Face]/*.wav
- `-model_path` Path to saved model file
- `-face` FACE Singal from which face on Cardboard HMD to train/test, accept "F", "R", or "L", default is `R`
- `-opt` Optimizer, support 'Adam' and 'SGD', default is `Adam`
- `-loss`  loss function, default is `categorical_crossentropy`
- `-lr` Learning rate, default is `1e-5`
- `-is` INPUT_SIZE [INPUT_SIZE ...] network input shape, default is `224, 224`
- `-e`  Training epochs, default is `100
- `-bs` Batch size, default is `8`
- `-nb_class` Number of classes. Default was calculated from the input training data label.
- `-model` Model to train, support "dn_121", "dn_169", "dn_201", "mobilenet", "vgg16", "vgg19", "resnet50", "resnet101" default is `dn_121`
- `-init` Initializer, default is `random_normal`
- `-aug` Data augmentation rate, default is `0.5`
-  `-save_to` Path to save the model
- `-mono`         Load .wav as mono, default is `False`
- `-l2m`         Load all data to memory, default is `False`
- `-train `       Training mode, default is `False`
- `-test`         Testing mode, default is `False`
- `-val_test`       Test on 8-2 split, default is `False`
- `-train_users`  TRAIN_USERS [TRAIN_USERS ...] User_ID list for training. Default using all users
- `-test_users` TEST_USERS [TEST_USERS ...] User_ID list for testing. Default using all users
- `-val_users` VAL_USERS [VAL_USERS ...] User_ID list for validating. Default validation set is 0.2 subset of training data



## Record your own data

You can make your own dataset using [our data collection tool](https://github.com/taizhouchen/GestOnHMD_DataCollection). 