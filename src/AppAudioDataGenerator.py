import numpy as np
import os
import skimage
from random import sample

class AppAudioDataGenerator():
    
    def __init__(self,
                 batch_size,
                 input_size,
                 output_size,
                 directory=None,
                 shuffle=False,
                 sample_size=None,
                 name='prediction',
                 shorten_factor=1,
                 mel_gamma=1):

        self.batch_size = batch_size

        self.input_size = np.array(input_size)

        self.image_height = output_size[0]
        self.image_width = output_size[1]

        self.shuffle = shuffle

        self.sample_size = sample_size

        self.shorten_factor = shorten_factor

        self.input_size[1] = self.input_size[1]//shorten_factor

        self.dir = directory

        self.mel_gamma = mel_gamma

        if self.dir != None:
            self.files = self.__get_files_from_directory()
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

    def __getitem__(self, index=0, data=None, num_tiles=None, return_filename=False):
        
        if type(data).__module__ == np.__name__:
            batch = data
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
            # new_X = np.empty((self.batch_size, self.image_height, self.image_width, 1))

            # slice_size = (original_width - self.image_width) // (64 - 1)
            
            # for idx, img in enumerate(X):
            #     all_tiles=[]
            #     for i in range(64):
            #         all_tiles.append(img[:,i*slice_size:(i*slice_size)+self.image_width,:])
            #     new_X[idx] = np.rot90(np.array(all_tiles).mean(axis=2), 3)
            
            # X = new_X
            # y = X
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

        #add gamma factor to increase/decrease contrast in mel spectrogram
        X = X**self.mel_gamma

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
            X[i,] = mel
            
        y = X

        return X, y

    def take(self, index, num_tiles=None, return_filename=False):
        return self.__getitem__(index=index, num_tiles=num_tiles, return_filename=return_filename)

    def get_tensors_from_data(self, data, num_tiles):
        return self.__getitem__(data=data, num_tiles=num_tiles)
