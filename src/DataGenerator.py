import numpy as np
import os
import tensorflow as tf
import requests
import librosa
from skimage.transform import resize
import streamlit as st


class AppDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                image_size,
                output_size=None,
                ):
        self._image_size = image_size
        self._img_height = image_size[0]
        self._img_width = image_size[1]

        self.output_size = output_size

    def __getitem__(self, index=0, return_filename=False, num_tiles=4, image_data=None, filename=None):
        batch = [filename]
        
        X, y = self.__get_data(batch, image_data)

        if self.output_size != None:
            
            if num_tiles > 1: 
                slice_size = (self._img_width - self.output_size[1]) // (num_tiles - 1)
            else:
                slice_size = 0

            all_tiles = []
            new_batch = []
            for idx, img in enumerate(X):
                for i in range(num_tiles):
                    all_tiles.append(img[:,i*slice_size:(i*slice_size)+self.output_size[1],:])
                    new_batch.append(batch[idx])
                        
            X = np.array(all_tiles)
            y = X
            
            if return_filename:
                batch = new_batch

        if return_filename:
            return batch, X, y
        else:
            return X, y

        
    def __get_data(self, batch, image_data=None):
        X = np.empty((1, self._img_height, self._img_width, 1))

        for i, file in enumerate(batch):
            scale = 1./255
            img = scale*np.array(image_data)
            X[i,] = tf.convert_to_tensor(img)
            
        y = X

        return X, y

    def get_vector_from_preview_link(self, track_url, track_id, num_tiles=32):
        img = download_preview_with_url(track_url, track_id)
        return self.__getitem__(num_tiles=num_tiles, image_data=img, filename=track_id)



def download_preview_with_url(track_url, track_id):

    preview = requests.get(track_url)
    st.write(preview)
    filename = f'data/{track_id}.mp3'

    with open(filename, 'wb') as f:
        f.write(preview.content)

    mel_image = convert_audio_to_mel_spectrogram(filename)

    os.remove(filename)

    return mel_image


def convert_audio_to_mel_spectrogram(filepath_to_audio, image_size=(128,512), n_mels=128, fmax=8000,):
    
    signal, sr = librosa.load(filepath_to_audio)
    
    mels = librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, fmax=fmax), ref=np.max)
    mel_image = (((80+mels)/80)*255)
    mel_image = np.flip(mel_image, axis=0)
    mel_image = resize(mel_image, (128,512)).astype(np.uint8)

    mel_image = np.expand_dims(mel_image, axis=2)

    return mel_image
