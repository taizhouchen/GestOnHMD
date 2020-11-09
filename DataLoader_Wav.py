import cv2
import numpy as np
import os
import io
import glob
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

from Utils import binary_onehot, shuffle
from sklearn.model_selection import train_test_split
from tensorflow_addons.image import sparse_image_warp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# warnings.filterwarnings('default')

class DataLoader(object):
    
    def __init__(self, path, batch_size, face, sr=44100, n_fft=2048, input_size=(224, 224), augment_rate=0.5, mono=False, train_users=None, val_users=None):
        self.path = path    # path to data folder, dir hierarchy should be: data/[user_ID]/[Face]/*
        self.batch_size = batch_size
        self.face = face    # L, R, F_L, F_R'
        self.sr = sr
        self.n_fft = n_fft
        self.input_size = input_size
        self.current_train_index = 0
        self.current_test_index = 0
        self.augment_rate = augment_rate
        self.mono = mono
        self.train_users = train_users
        self.val_users = val_users
        self.noise_list = ['noise' + os.path.sep + 'office.mp3', 'noise' + os.path.sep + 'street.mp3']
        self.noise_ = [librosa.load(noise_file, sr=self.sr, mono=self.mono)[0] for noise_file in self.noise_list]
        self.offset = 2000  # head and tail offset, in frames

        VALID_FACE = ['L', 'R', 'F']
        if self.face not in VALID_FACE:
            print('Invalid face: {}'.format(self.face))
            exit()
        
        all_files = glob.glob(os.path.join(self.path, '**' + os.path.sep + '*.wav'), recursive=True)
        
        self.files = [f for f in all_files if f.split(os.path.sep)[-2] == self.face]
        

        if self.train_users is not None:
            self.files = [f for f in self.files if f.split(os.path.sep)[-3] in self.train_users]
        self.labels = [f.split(os.path.sep)[-1].split('_')[0] for f in self.files]

        if len(self.files) == 0 or len(self.labels) == 0:
            print('Cannot find valid data.')
            exit()

        self.files, self.labels = shuffle(self.files, self.labels)
        
        self.nb_class = len(set(self.labels))

        if self.val_users is not None:
            files_test = [f for f in all_files if f.split(os.path.sep)[-2] == self.face and f.split(os.path.sep)[-3] in self.val_users]
            label_test = [f.split(os.path.sep)[-1].split('_')[0] for f in files_test]
            if len(self.files) == 0 or len(self.labels) == 0:
                print('Cannot find valid test data.')
                exit()
            self.X_test, self.y_test = shuffle(files_test, label_test)
            self.X_train = self.files
            self.y_train = self.labels
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.files, self.labels, test_size=0.2, random_state=42)
    
    def getTrainNextBatch(self):
        
        next_batch_index = self.current_train_index + self.batch_size

        images = []
        labels = []
        for i in range(self.current_train_index, next_batch_index):
            while True:
                y, sr = librosa.load(self.X_train[i], self.sr, mono=self.mono)

                # remove head and tail
                if self.mono:
                    y = y[self.offset:y.shape[0]-self.offset]
                else:
                    y = y[:, self.offset:y.shape[0]-self.offset]

                dice = random.random()
                if dice < self.augment_rate:
                    y = self.addNoise(y)
                fig = self.to_mels_fig(y, sr, self.augment_rate)
                if fig is not None: # may cause InvalidArgumentError exception
                    img = self.get_img_from_fig(fig)

                    # cv2.imshow('1', img)
                    # cv2.waitKey()
                    # cv2.destroyWindow('1')

                    # print(np.count_nonzero(img))
                    # print(1 * (img.shape[0] * img.shape[1] * img.shape[2]) // 2)

                    if self.mono:
                        no_zero_thrd = 1 * (img.shape[0] * img.shape[1] * img.shape[2]) // 2
                    else:
                        no_zero_thrd = 1 * (img.shape[0] * img.shape[1] * img.shape[2]) // 4

                    if np.count_nonzero(img) > no_zero_thrd:
                        break


            img = cv2.resize(img, self.input_size)

            images.append(img)
            labels.append(self.y_train[i])

        labels = binary_onehot(labels, self.nb_class)
        images = np.asarray(images)

        if next_batch_index + self.batch_size > len(self.X_train):
            self.current_train_index = 0
        else:
            self.current_train_index = next_batch_index
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train)

        return images, labels

    def to_mels_fig(self, y, sr, augment_rate = 0):
        """convert a audio sample sequence to melspectrogram
           if augment_rate > 0, then there is "augment_rate" chance to warp the melspectrogram in time axis
           and "augment_rate" * "augment_rate" change to do frequency masking

        Args:
            file_path ([str]): [path to a .wav file]

        Returns:
            [type]: [plt fig]
        """
        
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        if self.mono:
            melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=128, n_mels=256)
        else:
            melspectrogram_left = librosa.feature.melspectrogram(y=np.asfortranarray(y[0]), sr=sr, n_fft=self.n_fft, hop_length=128, n_mels=256)  
            melspectrogram_right = librosa.feature.melspectrogram(y=np.asfortranarray(y[1]), sr=sr, n_fft=self.n_fft, hop_length=128, n_mels=256) 

            melspectrogram = np.concatenate((melspectrogram_left, melspectrogram_right)) 

        dice = random.random()
        if dice < augment_rate:
            # print('Augmenting data...')
            melspectrogram = np.transpose(melspectrogram)
            shape = melspectrogram.shape
            melspectrogram = np.reshape(melspectrogram, (-1, shape[0], shape[1], 1))

            if random.random() < augment_rate: # dice twice to decide if masking
                melspectrogram = self.frequency_masking(melspectrogram, shape[0])

            try:
                melspectrogram = self.sparse_warp(melspectrogram, time_warping_para=50)
                S_dB = librosa.power_to_db(np.transpose(melspectrogram[0, :, :, 0]), ref=np.max)
                librosa.display.specshow(S_dB, ax=ax, sr=sr)
                ax.set_axis_off()
                return fig

            except tf.python.framework.errors_impl.InvalidArgumentError:
                # print('except tf.python.framework.errors_impl.InvalidArgumentError')
                return None
        else:
            S_dB = librosa.power_to_db(melspectrogram, ref=np.max)
            librosa.display.specshow(S_dB, ax=ax, sr=sr)
            ax.set_axis_off()

        return fig


    def get_img_from_fig(self, fig, dpi=180):
        """convert a plt fig to opencv mat

        Args:
            fig ([type]): [description]
            dpi (int, optional): [description]. Defaults to 180.

        Returns:
            [type]: [description]
        """

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches = 'tight', pad_inches = 0)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)

        return img


    def getTestNextBatch(self):

        next_batch_index = self.current_test_index + self.batch_size

        images = []
        labels = []
        for i in range(self.current_test_index, next_batch_index):
            y, sr = librosa.load(self.X_test[i], self.sr, mono=self.mono)
            img = self.get_img_from_fig(self.to_mels_fig(y, sr))
            img = cv2.resize(img, self.input_size)
            images.append(img)
            labels.append(self.y_test[i])

        labels = binary_onehot(labels, self.nb_class)
        images = np.asarray(images)

        if next_batch_index + self.batch_size > len(self.X_test):
            self.current_test_index = 0
        else:
            self.current_test_index = next_batch_index
            self.X_test, self.y_test = shuffle(self.X_test, self.y_test)

        return images, labels

    def getAllInMemory(self):
        """load all data in memory for training

        Returns:
            [type]: [description]
        """

        train_images = []
        for x in self.X_train:
            y, sr = librosa.load(x, self.sr, mono=self.mono)
            # remove head and tail
            if self.mono:
                y = y[self.offset:y.shape[0]-self.offset]
            else:
                y = y[:, self.offset:y.shape[0]-self.offset]
            img = self.get_img_from_fig(self.to_mels_fig(y, sr))
            img = cv2.resize(img, self.input_size)
            train_images.append(img)

        train_labels = binary_onehot(self.y_train, self.nb_class)
        train_images = np.asarray(train_images)

        test_images = []
        for x in self.X_test:
            y, sr = librosa.load(x, self.sr, mono=self.mono)
            # remove head and tail
            if self.mono:
                y = y[self.offset:y.shape[0]-self.offset]
            else:
                y = y[:, self.offset:y.shape[0]-self.offset]
            img = self.get_img_from_fig(self.to_mels_fig(y, sr))
            img = cv2.resize(img, self.input_size)
            test_images.append(img)

        test_labels = binary_onehot(self.y_test, self.nb_class)
        test_images = np.asarray(test_images)

        return train_images, test_images, train_labels, test_labels

    def getAllTestInMemory(self):

        test_images = []
        print('loading {} data'.format(len(self.X_test)))
        for x in self.X_test:
            y, sr = librosa.load(x, self.sr, mono=self.mono)
            # remove head and tail
            if self.mono:
                y = y[self.offset:y.shape[0]-self.offset]
            else:
                y = y[:, self.offset:y.shape[0]-self.offset]
            img = self.get_img_from_fig(self.to_mels_fig(y, sr))
            img = cv2.resize(img, self.input_size)
            test_images.append(img)

        test_labels = binary_onehot(self.y_test, self.nb_class)
        test_images = np.asarray(test_images)

        return test_images, test_labels


    def getAllDataInMemory(self):
        """load all data in memory for testing

        Returns:
            [type]: [description]
        """

        print('Loading data...')
        train_images, test_images, train_labels, test_labels = self.getAllInMemory()
        if len(train_images) != 0 and len(test_images) == 0:
            return train_images, train_labels
        elif len(train_images) == 0 and len(test_images) != 0:
            return test_images, test_labels
        else:
            return np.concatenate((train_images, test_images)), np.concatenate((train_labels, test_labels))
        

    def sparse_warp(self, mel_spectrogram, time_warping_para=80):
        """Spec augmentation Calculation Function.

        # Arguments:
        mel_spectrogram(numpy array): audio file path of you want to warping and masking.
        time_warping_para(float): Augmentation parameter, "time warp parameter W".
            If none, default = 80 for LibriSpeech.

        # Returns
        mel_spectrogram(numpy array): warped and masked mel spectrogram.
        """

        fbank_size = tf.shape(mel_spectrogram)
        n, v = fbank_size[1], fbank_size[2]

        # Image warping control point setting.
        # Source
        pt = tf.random.uniform([], time_warping_para, n-time_warping_para, tf.int32) # radnom point along the time axis
        src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
        src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
        src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
        src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

        # Destination
        w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
        dest_ctr_pt_freq = src_ctr_pt_freq
        dest_ctr_pt_time = src_ctr_pt_time + w
        if tf.math.reduce_any(dest_ctr_pt_time < 0):
            dest_ctr_pt_time = tf.repeat(0, v)
        if tf.math.reduce_any(dest_ctr_pt_time > n):
            dest_ctr_pt_time = tf.repeat(n, v)
        dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
        dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

        # warp
        source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
        dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

        warped_image, _ = sparse_image_warp(mel_spectrogram,
                                            source_control_point_locations,
                                            dest_control_point_locations)
        return warped_image


    def frequency_masking(self, mel_spectrogram, v, frequency_masking_para=27, frequency_mask_num=2):
        """Spec augmentation Calculation Function.

        # Arguments:
        mel_spectrogram(numpy array): audio file path of you want to warping and masking.
        frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
            If none, default = 100 for LibriSpeech.
        frequency_mask_num(float): number of frequency masking lines, "m_F".
            If none, default = 1 for LibriSpeech.

        # Returns
        mel_spectrogram(numpy array): warped and masked mel spectrogram.
        """
        fbank_size = tf.shape(mel_spectrogram)
        n, v = fbank_size[1], fbank_size[2]

        for i in range(frequency_mask_num):
            f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
            v = tf.cast(v, dtype=tf.int32)
            f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

            # warped_mel_spectrogram[f0:f0 + f, :] = 0
            mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                            tf.zeros(shape=(1, n, f, 1)),
                            tf.ones(shape=(1, n, f0, 1)),
                            ), 2)
            mel_spectrogram = mel_spectrogram * mask
        return tf.cast(mel_spectrogram, dtype=tf.float32)

    def addNoise(self, y, noise_weight=0.1):

        noise = random.choice(self.noise_)
        noise = np.asarray(noise)

        offset = round(random.random() * (noise.shape[-1] - y.shape[-1]))

        duration = y.shape[-1]

        # print('noise shape:')
        # print(noise.shape)
        # print('y shape:')
        # print(y.shape)
        # print('offset')
        # print(offset)
        # print('duration')
        # print(duration)

        if self.mono:
            # y = y[0:m]
            y_n = noise[offset:offset+duration]
        else:
            # y = y[:, 0:m]
            y_n = noise[:, offset:offset+duration]

        new_y = y * (1-noise_weight) + y_n * noise_weight

        # librosa.output.write_wav('add_noise.wav', new_y, self.sr)

        # print(y_n.shape)
        # print(new_y.shape)

        return new_y



    def batchGenerator(self, type_):
        if type_ == 'train':
            while True:
                yield self.getTrainNextBatch()
        if type_ == 'test':
            while True:
                yield self.getTestNextBatch()

    def getClassNum(self):
        return self.nb_class

    def setClassNum(self, nb_class):
        self.nb_class = nb_class

    def getTrainDataLength(self):
        return len(self.X_train)

    def getTestDataLength(self):
        return len(self.X_test)

    def printSummary(self):
        print('=' * 20)
        print("Number of class: " + str(self.getClassNum()))
        print("Training data length: " + str(self.getTrainDataLength()))
        print("Testing data length: " + str(self.getTestDataLength()))
 


if __name__ == "__main__":
    mDataLoader = DataLoader('./data_new', 8, 'R', mono=False)
    gen = mDataLoader.batchGenerator('train')
    while True:
        image, label = next(gen)
        print(image.shape)
        print(label.shape)