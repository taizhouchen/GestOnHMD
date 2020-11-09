import keras
import argparse
import os
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from DataLoader_Wav import DataLoader
import time


# from models.mobilenet_v2 import MobileNetv2
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', dest='data_path', default='./data', type=str,
                        help='Path to data dictionary. A valid data dictionary hierarchy is: data/[user_ID]/[Face]/*.wav')
    parser.add_argument('-model_path', dest='model_path', default=None, type=str,
                        help='Path to saved model file')
    parser.add_argument('-face', dest='face', default='R', type=str,
                        help='Singal from which face on Cardboard HMD to train/test')
    parser.add_argument('-opt', dest='optimizer', default='Adam', type=str,
                        help='Optimizer, support \'Adam\' and \'SGD\'')
    parser.add_argument('-loss', dest='loss', default='categorical_crossentropy', type=str,
                        help='loss function')
    parser.add_argument('-lr', dest='lr', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('-is', dest='input_size', nargs='+', default=[224, 224], type=int, help='network input shape')
    parser.add_argument('-e', dest='epochs', default=1000, type=int, help='Training epochs')
    parser.add_argument('-bs', dest='batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('-nb_class', dest='nb_class', default=None, type=int, help='Number of classes. Default was calculated from the input training data label.')
    parser.add_argument('-model', dest='model_name', default='DN_121', type=str, help='Model to train')
    parser.add_argument('-init', dest='initializer', default='random_normal', type=str, help='Initializer')
    parser.add_argument('-aug', dest='aug', default=0.5, type=float, help='Data augmentation rate')
    parser.add_argument('-save_to', dest='checkpoint_dir', default=None, type=str, help='Path to save the model')
    parser.add_argument('-mono', dest='mono', default=False, action='store_true', 
                        help='Load .wav as mono')
    parser.add_argument('-l2m', dest='load_to_memory', default=False, action='store_true', 
                        help='Load all data to memory')
    parser.add_argument('-train', dest='is_training', default=False, action='store_true', 
                        help='Training mode')
    parser.add_argument('-test', dest='is_testing', default=False, action='store_true', 
                        help='Testing mode')
    parser.add_argument('-val_test', dest='val_test', default=False, action='store_true', 
                        help='Test on 8-2 split')
    parser.add_argument('-train_users', dest='train_users', nargs='+', type=str, default=None, help='User_ID list for training. Default using all users')
    parser.add_argument('-test_users', dest='test_users', nargs='+', type=str, default=None, help='User_ID list for testing. Default using all users')
    parser.add_argument('-val_users', dest='val_users', nargs='+', type=str, default=None, help='User_ID list for validating. Default validation set is 0.2 subset of training data')

    return parser.parse_args()

def train(data_path, model_path, model_name, face, input_size, mono, load_to_memory, aug, batch_size, 
    epochs, lr, optimizer, loss, initializer, checkpointDir, train_users, val_users):
    
    mDataLoader = DataLoader(data_path, batch_size, face, input_size=input_size, augment_rate=aug, mono=mono, train_users=train_users, val_users=val_users)
    
    mDataLoader.printSummary()
    nb_class = mDataLoader.getClassNum()
    input_size = (input_size[0], input_size[1],3)

    model = None

    # Build the model
    if model_name.lower() == 'dn_121':
        # from models.densenet121 import DenseNet
        # model = DenseNet(reduction=0.5, classes=nb_class, weights_path=model_path)
        model = keras.applications.DenseNet121(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'dn_169':
        # from models.densenet169 import DenseNet
        # model = DenseNet(reduction=0.5, classes=nb_class, weights_path=model_path)
        model = keras.applications.DenseNet169(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'dn_201':
        model = keras.applications.DenseNet201(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'mobilenet':
        model = keras.applications.MobileNetV2(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'vgg16':
        model = keras.applications.VGG16(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'vgg19':
        model = keras.applications.VGG19(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'resnet50':
        model = keras.applications.ResNet50V2(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'resnet101':
        model = keras.applications.ResNet101V2(input_shape=input_size, classes=nb_class, weights=model_path)

    model.summary()

    tensorboard = TensorBoard(log_dir=os.path.join('logs', model_name + '_' + face))

    if checkpointDir == None:
        checkpointDir = './checkpoints/' + model_name + '_' + face
    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)
        print('New dir: ' + checkpointDir + ' was created')

    file_path = os.path.join(checkpointDir, 'model.h5')
    checkpointer = ModelCheckpoint(filepath=file_path, verbose=1, 
        monitor='val_accuracy', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, 
        verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    if optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    else:
        opt = keras.optimizers.SGD(learning_rate=lr)

    METRICS = [
      'accuracy',
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall')
    ]

    model.compile(loss=loss, optimizer=opt,
        metrics=METRICS)

    if load_to_memory:
        X_train, X_test, y_train, y_test = mDataLoader.getAllInMemory()
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[tensorboard, checkpointer],
            epochs=epochs
        )
    else:
        train_generator = mDataLoader.batchGenerator('train')
        val_generator = mDataLoader.batchGenerator('test')
        step_per_epoch = mDataLoader.getTrainDataLength() // batch_size
        validation_steps = mDataLoader.getTestDataLength() // batch_size
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=step_per_epoch,
            epochs=epochs,
            callbacks=[tensorboard, checkpointer],
            validation_data=val_generator,
            validation_steps=validation_steps
        )


def test(data_path, model_name, face, model_path, input_size, mono, nb_class, test_users, val_test):

    if model_path is None:
        print('No model to test.')
        exit()

    tic = time.time()
    mDataLoader = DataLoader(data_path, None, face, input_size=input_size, mono=mono, train_users=test_users)

    if nb_class is None:
        nb_class = mDataLoader.getClassNum()
    else:
        mDataLoader.setClassNum(nb_class)

    model = None
    input_size = (input_size[0], input_size[1],3)

    # Build the model
    if model_name.lower() == 'dn_121':
        # from models.densenet121 import DenseNet
        # model = DenseNet(reduction=0.5, classes=nb_class, weights_path=model_path)
        model = keras.applications.DenseNet121(input_shape=input_size, classes=nb_class, weights=model_path)
    # if model_name.lower() == 'dn_161':
    #     from models.densenet161 import DenseNet
    #     model = DenseNet(reduction=0.5, classes=nb_class, weights_path=model_path)
    if model_name.lower() == 'dn_169':
        # from models.densenet169 import DenseNet
        # model = DenseNet(reduction=0.5, classes=nb_class, weights_path=model_path)
        model = keras.applications.DenseNet169(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'dn_201':
        model = keras.applications.DenseNet201(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'mobilenet':
        model = keras.applications.MobileNetV2(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'vgg16':
        model = keras.applications.VGG16(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'vgg19':
        model = keras.applications.VGG19(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'resnet50':
        model = keras.applications.ResNet50V2(input_shape=input_size, classes=nb_class, weights=model_path)
    if model_name.lower() == 'resnet101':
        model = keras.applications.ResNet101V2(input_shape=input_size, classes=nb_class, weights=model_path)
    
    model.summary()

    if val_test:
        print('loading test set...')
        X, y = mDataLoader.getAllTestInMemory()
    else:
        X, y = mDataLoader.getAllDataInMemory()
    print('Testing on ' + str(X.shape[0]) + ' data')
    print('Loading data and model time: {}'.format(time.time() - tic))

    tic = time.time()
    predicted_labels = np.argmax(model.predict(X), axis=-1)
    print('Inference time: {}'.format(time.time() - tic))

    
    
    y = np.argmax(y, axis=1)
    accuracy = accuracy_score(y, predicted_labels)

    print('Accuracy: {}'.format(accuracy))
    print(classification_report(y, predicted_labels, digits=4))

    fig = plt.Figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    cm = confusion_matrix(y, predicted_labels)
    print(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm_normalized)
    sns.heatmap(cm_normalized, ax=ax, annot=True, cmap='Blues', fmt='.2f')

    result_path = 'results'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    file_name = model_name + '_' + face + '_' + str(round(accuracy, 4)) + '.png'
    path = os.path.join(result_path, file_name)

    fig.savefig(path, format='png', dpi=300, quality=95)
    print('Detailed results image was saved to {}'.format(path))


def main():
    args = get_args_parser()
    print(args)

    VALID_MODEL = ['dn_121', 'dn_169', 'dn_201', 'mobilenet', 'vgg16', 'vgg19', 'resnet50', 'resnet101']

    data_path = args.data_path
    model_name = args.model_name
    face = args.face
    input_size = tuple(args.input_size)
    load_to_memory = args.load_to_memory
    batch_size = args.batch_size
    nb_class = args.nb_class
    epochs = args.epochs
    lr = args.lr
    aug = args.aug
    optimizer = args.optimizer
    loss = args.loss
    initializer = args.initializer
    is_training = args.is_training
    is_testing = args.is_testing
    model_path = args.model_path
    checkpointDir = args.checkpoint_dir
    mono = args.mono
    train_users = args.train_users
    val_users = args.val_users
    test_users = args.test_users
    val_test = args.val_test

    if not (is_training or is_testing):
        print('Unknown mode. Please assign value to -train or -test')
        exit()

    if model_name.lower() not in VALID_MODEL:
        print('Unknown model.')
        exit()

    if is_training:
        train(data_path, model_path, model_name, face, input_size, mono, load_to_memory, aug, 
                batch_size, epochs, lr, optimizer, loss, initializer, checkpointDir, train_users, val_users)
    if is_testing:
        test(data_path, model_name, face, model_path, input_size, mono, nb_class, test_users, val_test)

if __name__ == "__main__":
    main()