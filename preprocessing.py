import os
from random import shuffle
import random
import tensorflow as tf
import glob
from config import config
import pandas as pd
#dont import other libraries

class Preprocessing:
    def __init__(self):
        print('preprocessing instance creation started')

        self.train_tracks = pd.read_parquet(config['train_dataset_path'])
        self.test_tracks = pd.read_parquet(config['test_dataset_path'])

        self.input_len = config['input_len']
        self.sr = config['sample_rate']
        
    def create_iterators(self):
        # get the filenames split into train test validation
        train_files = self.get_files_from_df(self.train_tracks)
        test_files = self.get_files_from_df(self.test_tracks)

        shuffle(train_files)
        # get the commands and some prints
        self.genres = self.get_genres()
        self.num_classes = len(self.genres)
        print('len(train_data)', len(train_files))
        print('prelen(test_data)', len(test_files))
        print('genres: ', self.genres)
        print('number of genres: ', len(self.genres))

        # make tf dataset object
        self.train_dataset = self.make_tf_dataset_from_list(train_files)
        self.test_dataset = self.make_tf_dataset_from_list(test_files, is_validation=True)

    def get_files_from_df(self, df):
        files = [[x[1], str(x[2]), str(x[3]), x[0]] for x in df.values]
        
        return files

    def get_genres(self):
        genres = tf.convert_to_tensor(self.train_tracks['genre'].unique().tolist())
        return genres

    
    def get_label(self, label_raw):
        label = tf.argmax(label_raw == self.genres)

        return label

    def make_tf_dataset_from_list(self, filenames_list, is_validation = False):
        """
        ARGS:
            filenames_list is a list of file_paths
            is_validation is a boolean which should be true when makeing val_dataset

        Using the list create tf.data.Dataset object
        do necessary mappings (methods starting with 'map'),
        use prefetch, shuffle, batch methods
        bonus points` mix with background noise 
        """
        dataset = tf.data.Dataset.from_tensor_slices(filenames_list).map(self.map_get_waveform_and_label).map(self.map_add_padding)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.batch(config['train_params']['batch_size'])
        if not is_validation:
            dataset = dataset.shuffle(1000)
            #.cache()
        
        return dataset

    def map_get_waveform_and_label(self, file_path_pair):
        """
        Map function
        for every filepath return its waveform (use only tensorflow) and label 
        """
        label = self.get_label(file_path_pair[3])
        data_str = tf.io.read_file(file_path_pair[0])
        waveform, _ = tf.audio.decode_wav(data_str)
        waveform = tf.squeeze(waveform, axis=1)
        print('wavform', tf.shape(waveform))
        s = int(file_path_pair[1])*self.sr
        e = int(file_path_pair[2])*self.sr
        
        segment = waveform[s:e]
        print('segment', tf.shape(segment))

        return segment, label

    def map_add_padding(self, audio, label):
        return [self.add_paddings(audio), label]

    def add_paddings(self, wav):
        """
        all the data should be 2 seconds (16000 points)
        pad with zeros to make every wavs lenght 16000 if needed.
        """
        # padded_wav = tf.pad(wav, [[tf.random.uniform(shape=(), minval=0, maxval=(self.input_len - tf.shape(wav)[0]), dtype=tf.int32), 0]], constant_values=0)
        # padded_wav = tf.pad(padded_wav, [[0, self.input_len - tf.shape(padded_wav)[0]]], constant_values=0)
        print('before pad wav', tf.shape(wav))
        padded_wav = tf.pad(wav, [[self.input_len - tf.shape(wav)[0], 0]], constant_values=0)
        print('padded_wav', tf.shape(padded_wav))
        return padded_wav
