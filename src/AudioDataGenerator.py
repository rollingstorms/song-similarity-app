import tensorflow as tf
import numpy as np
import os
import skimage
from random import shuffle, sample

class AudioDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,
                 batch_size,
                 input_size,
                 output_size,
                 directory,
                 shuffle=False,
                 sample_size=None,
                 train_test_split=False,
                 test_size=.2,
                 name='prediction',
                 file_list=None,
                 shorten_factor=1):

        self.batch_size = batch_size

        self.input_size = np.array(input_size)

        self.image_height = output_size[0]
        self.image_width = output_size[1]

        self.dir = directory

        self.shuffle = shuffle

        self.sample_size = sample_size
        
        self.test_size = test_size

        self.shorten_factor = shorten_factor

        self.input_size[1] = self.input_size[1]//shorten_factor
        

        if file_list == None:
            if self.dir != None:
                self.files = self.__get_files_from_directory()
        else:
            try:
                self.files = self.__collect_npy_files(file_list)
            except TypeError:
                print('file_list is not a list')

        if train_test_split:
            self.__train_test_split(self.files)

        else:
            print(f'Found {len(self.files)} files for {name} set')

        self.size = len(self.files)

    def __get_files_from_directory(self):
        files = os.listdir(self.dir)

        files = self.__collect_npy_files(files)

        if self.shuffle:
            shuffle(files)
        
        if self.sample_size != None:
            files = sample(files, self.sample_size)

        return files

    def __collect_npy_files(self, files):
        filetypes = ['npy']
        return [file for file in files if file.split('.')[-1] in filetypes]

    def __train_test_split(self, files):
        
        if self.shuffle:
            shuffle(files)
            
        file_list_length = len(files)
        test_split = int(file_list_length * (1 - self.test_size))
        
        train_files = files[:test_split]
        test_files = files[test_split:]
        
        self.train = AudioDataGenerator(batch_size=self.batch_size,
                                        input_size=self.input_size,
                                        output_size=(self.image_height, self.image_width),
                                        directory=self.dir,
                                        shuffle=self.shuffle,
                                        name='training',
                                        file_list=train_files,
                                        shorten_factor=self.shorten_factor)
        
        self.test = AudioDataGenerator(batch_size=self.batch_size,
                                       input_size=self.input_size,
                                       output_size=(self.image_height, self.image_width),
                                       directory=self.dir,
                                       shuffle=self.shuffle,
                                       name='testing',
                                       file_list=test_files,
                                       shorten_factor=self.shorten_factor)        

    def __len__(self):
        return len(self.files) // self.batch_size

    def __getitem__(self, index=0, data=None, num_tiles=None, return_filename=False):
        
        if type(data).__module__ == np.__name__:
            batch = tf.convert_to_tensor(data)
            X = np.expand_dims(batch, axis=0)
            X = np.expand_dims(X, axis=3)
            y = X
        elif data == None:
            batch = self.files[index*self.batch_size:index*self.batch_size+self.batch_size]
            X, y = self.__getdata(batch)
        else:
            raise TypeError('data must be a np.array or "None"')


        original_height = X.shape[1]
        original_width = X.shape[2]

        if num_tiles == None:
            if self.image_width < original_width:
                rand_x_index = np.random.randint(low=0, high=original_width - self.image_width)
            else:
                rand_x_index = 0

            X = X[:,0:self.image_height,rand_x_index:rand_x_index+self.image_width,:]
            y = X
        else:
            if num_tiles > 1: 
                slice_size = (original_width - self.image_width) // (num_tiles - 1)
            else:
                slice_size = 0

            all_tiles = []
            new_batch = []
            for idx, img in enumerate(X):
                for i in range(num_tiles):
                    all_tiles.append(img[:,i*slice_size:(i*slice_size)+self.image_width,:])
                    new_batch.append(batch[idx])
                        
            X = np.array(all_tiles)
            y = X
        if return_filename:
            return X, y, batch
        else:
            return X, y

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.files)


    def __getdata(self, batch):

        X = np.empty((self.batch_size, self.input_size[0], self.input_size[1], 1))

        for i, file in enumerate(batch):
            mel = np.load(self.dir + '/' + file, allow_pickle=True)
            if self.shorten_factor != 1:
                mel = skimage.transform.resize(mel, (self.input_size[0], self.input_size[1]))
            if mel.shape[1] < self.input_size[1]:
                mel = skimage.transform.resize(mel, (self.input_size[0], self.input_size[1]))
            mel = np.expand_dims(mel, axis=2)
            X[i,] = tf.convert_to_tensor(mel)
            
        y = X

        return X, y

    def take(self, index, num_tiles=None, return_filename=False):
        return self.__getitem__(index=index, num_tiles=num_tiles, return_filename=return_filename)

    def get_tensors_from_data(self, data, num_tiles):
        return self.__getitem__(data=data, num_tiles=num_tiles)

