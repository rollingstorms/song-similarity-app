from src.audio_functions import *
import numpy as np
import requests
import streamlit as st
import pandas as pd
from pyarrow import feather
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from joblib import load, dump
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os



class AudioSpectrogramConverter:
    
    def __init__(self,
                 save_folder,
                 fft_size=2048,
                 spec_thresh=4,
                 n_mel_freq_components=64,
                 shorten_factor=10,
                 start_freq=20,
                 end_freq=16000):

        self.save_folder = save_folder

        self.fft_size = fft_size  # window size for the FFT
        self.step_size = fft_size // 16  # distance to slide along the window (in time)
        self.spec_thresh = spec_thresh  # threshold for spectrograms (lower filters out more noise)
        # For mels
        self.n_mel_freq_components = n_mel_freq_components # number of mel frequency channels
        self.shorten_factor = shorten_factor  # how much should we compress the x-axis (time)
        self.start_freq = start_freq  # Hz # What frequency to start sampling our melS from
        self.end_freq = end_freq  # Hz # What frequency to stop sampling our melS from


        self.mel_filter, self.mel_inversion_filter = create_mel_filter(
            fft_size=self.fft_size,
            n_freq_components=self.n_mel_freq_components,
            start_freq=self.start_freq,
            end_freq=self.end_freq,
        )

        create_new_directories(save_folder)
    
    def _download_link_to_file(self, link, filename):
        r = requests.get(link)

        with open(filename, 'wb') as f:
            f.write(r.content)
    
    def _convert_mp3_to_mel(self, file, scale=True, log=True):

        track = AudioSegment.from_mp3(file)

        samples, sample_rate = pydub_to_np(track)

        sample = np.mean(samples, axis=0)

        wav_spectrogram = pretty_spectrogram(
            samples[:,0].astype("float64"),
            fft_size=self.fft_size,
            step_size=self.step_size,
            log=log,
            thresh=self.spec_thresh,
            scale=scale
        )

        mel_spec = make_mel(wav_spectrogram, self.mel_filter, shorten_factor=self.shorten_factor)

        return mel_spec

    def _delete_audio_file(self, filename):
        os.remove(filename)

    def _save_spectrogram(self, mel, file_id):

        create_new_directories(self.save_folder + '/mels')

        npy_filename = self.save_folder + f'/mels/{file_id}.npy'
        np.save(npy_filename, mel)

    def convert(self, link, file_id=None, save=False, scale=True, log=False, save_mp3=False):

        create_new_directories(self.save_folder + '/mp3s')

        filename = self.save_folder + f'/mp3s/{file_id}.mp3'
        self._download_link_to_file(link, filename)
        mel = self._convert_mp3_to_mel(filename, scale=scale, log=log)
        if log == False:
            mel = np.maximum(mel, 1)
            mel = (20.0 * np.log10(np.maximum(1e-10, mel)))/50
            mel = np.minimum(mel, 1)

        if save_mp3 != True:
            self._delete_audio_file(filename)

        if np.isnan(mel.max()):
            return np.zeros_like(mel)

        if save:
            self._save_spectrogram(mel, file_id)
        
        return mel
    
class Sonufy:
    
    def __init__(self,
                 latent_dims,
                 output_size,
                 num_tiles=32,
                 fft_size=2048,
                 spec_thresh=1,
                 n_mel_freq_components=64,
                 shorten_factor=10,
                 start_freq=20,
                 end_freq=8000,
                 final_shorten_factor=1,
                 mel_gamma=1):

        self.latent_dims = latent_dims
        
        self.image_height = output_size[0]
        self.image_width = output_size[1]
        self.num_tiles = num_tiles

        self.fft_size = fft_size  # window size for the FFT
        self.spec_thresh = spec_thresh  # threshold for spectrograms (lower filters out more noise)
        # For mels
        self.n_mel_freq_components = n_mel_freq_components # number of mel frequency channels
        self.shorten_factor = shorten_factor  # how much should we compress the x-axis (time)
        self.start_freq = start_freq  # Hz # What frequency to start sampling our melS from
        self.end_freq = end_freq

        self.final_shorten_factor = final_shorten_factor

        self.mel_gamma = mel_gamma #mel gamma for increase/decrease contrast in mel spectrogram

        self.latent_cols = [f'latent_{i}' for i in range(self.latent_dims)]

        client_id = st.secrets['clientId']
        client_secret = st.secrets['clientSecret']

        credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

        self._spotify = spotipy.Spotify(client_credentials_manager=credentials_manager)

    def load_tracks_db(self, filename):

        self.all_tracks = feather.read_feather(filename)
        self.all_tracks = self.all_tracks[~self.all_tracks.track_preview_link.isna()].reset_index(drop=True)


    def download_links(self, save_folder, all_tracks_file):

        self.load_tracks_db(all_tracks_file)

        create_new_directories(save_folder)
        # try:
        already_downloaded = os.listdir(save_folder + '/mels')
        already_downloaded = [x for x in list(map(lambda x: x.split('.')[0] if x.split('.')[1] == 'npy' else None, already_downloaded)) if x != None]

        df = self.all_tracks[~self.all_tracks.track_id.isin(already_downloaded)]

        # except:
        #     pass

        self.asc = AudioSpectrogramConverter(save_folder=save_folder, 
                                        fft_size=self.fft_size,
                                        spec_thresh=self.spec_thresh,
                                        n_mel_freq_components=self.n_mel_freq_components,
                                        shorten_factor=self.shorten_factor,
                                        start_freq=self.start_freq,
                                        end_freq=self.end_freq)

        
        indices = df.index
        MAX_THREADS = 30
        threads = min(MAX_THREADS, len(indices))

        # with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        #     executor.map(self._download_link, indices)

        for i in indices:
            self._download_link(i)

    def _download_link(self, index):

        row = self.all_tracks.iloc[index]

        self.asc.convert(link=row.track_preview_link, file_id=row.track_id, save=True, log=False, scale=False)

        print(index, end='\r')

    
    def build_model(self, learning_rate=1e-3, filters=(16,32,64)):

        import tensorflow as tf

        #load model class
        from src.model import Time_Freq_Autoencoder

        #build/compile
        #save as self.autoencoder
        self.autoencoder = Time_Freq_Autoencoder(image_width=self.image_width, image_height=self.image_height, latent_dim=self.latent_dims, kernel_size=5, filters=filters)
        
        from tensorflow.keras.optimizers import Adam

        self.opt = Adam(learning_rate=learning_rate)

        self.autoencoder.compile(optimizer=self.opt, loss=tf.keras.losses.mse)

        self.autoencoder.build(input_shape=(None, self.image_height, self.image_width))

    def save_full_model(self, save_folder):

        create_new_directories(save_folder)

        self.autoencoder.save(save_folder)

        print(f'Model saved to "{save_folder}".')

    def load_full_model(self, save_folder):

        import tensorflow as tf

        self.autoencoder = tf.keras.models.load_model(save_folder)

        print(f'Autoencoder loaded from "{save_folder}".')
    
    def save_encoder(self, save_folder):

        import tensorflow as tf

        create_new_directories(save_folder)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.autoencoder.encoder)
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        quantized_tflite_model = converter.convert()

        with open(save_folder + '/encoder.tflite', 'wb') as f:
            f.write(quantized_tflite_model)

        print(f'Encoder saved to "{save_folder}".')

        self.load_encoder(save_folder)

    def load_encoder(self, save_folder, app=False):

        if app:

            import tflite_runtime.interpreter as tflite

            self.interpreter = tflite.Interpreter(save_folder+'/encoder.tflite')

        else:

            import tensorflow as tf

            self.interpreter = tf.lite.Interpreter(save_folder+'/encoder.tflite')


    def save(self, save_folder):
        try:
            self.save_full_model(save_folder)
            print('saved full model')
        except:
            print('failed to save full model')
        try:
            self.save_encoder(save_folder)
            print('saved encoder')
        except:
            print('failed to save encoder')
        try:
            self.save_db(save_folder)
            print('saved database files')
        except:
            print('failed to save database files')

        print(f'Saved full model, encoder, and database files in {save_folder}.')

    def load(self, save_folder):

        try:
            self.load_full_model(save_folder)
            print('loaded full model')
        except:
            print('failed to load full model')

        try:
            self.load_encoder(save_folder)
            print('loaded encoder')
        except:
            print('failed to load encoder')

        try:
            self.load_db(save_folder)
            print('loaded database files')
        except:
            print('failed to load database files')

    
    def build_vectors_from_model(self, mel_directory, all_tracks_file, sample_size=None):
        from src.AppAudioDataGenerator import AppAudioDataGenerator

        self.load_tracks_db(all_tracks_file)

        #create prediction generator
        input_size = self._get_input_shape(mel_directory)

        self.prediction_generator = AppAudioDataGenerator(batch_size=1, input_size=input_size, output_size=(self.image_height, self.image_width), directory=mel_directory, shuffle=False, sample_size=sample_size, shorten_factor=self.final_shorten_factor, mel_gamma=self.mel_gamma)
        
        print('Getting predictions from autoencoder...')
        start_time = time.time()
        #build tracks
        results = []
        for i in range(self.prediction_generator.size):
            latent_img, _, filename = self.prediction_generator.take(i, num_tiles=self.num_tiles, return_filename=True)
            # latent_img, _, filename = self.prediction_generator.take(i, num_tiles=None, return_filename=True)

            latent_space = self.run_inference(latent_img).mean(axis=0)
            # latent_space = self.run_inference(latent_img)[0]

            result={
                'track_id':filename[0].split('.')[0],
                'filename':filename[0],
                  }
            for idx, col in enumerate(latent_space):
                result[f'latent_{idx}'] = col

            results.append(result)

            progress_bar(i+1, self.prediction_generator.size)

        print('\n')
        print(round((time.time()-start_time)/60, 2),'minutes elapsed')

        start_time = time.time()
        print('Building tracks dataframe...')
        results_df = pd.DataFrame(results)
        
        track_latents = results_df.merge(self.all_tracks, how='left', left_on='track_id', right_on='track_id')
        track_latents = track_latents.drop_duplicates(subset='track_id')
        track_latents = track_latents.reset_index(drop=True)

        self.tracks = track_latents

        
        self._scaler = StandardScaler()
        track_latent_scaled = self._scaler.fit_transform(track_latents[self.latent_cols])
        track_latents[self.latent_cols] = track_latent_scaled.astype(np.float16)

        self.tracks = track_latents

        print(f'Track dataframe built. {round((time.time()-start_time)/60,2)} minutes elapsed')
        #build genre space from base genres
        start_time = time.time()
        print('Building genre distributions...')
        genre_rows = []
        for idx, row in self.tracks.iterrows():
            for genre in row.artist_genres:
                new_row = row
                new_row['genre'] = genre
                genre_rows.append(new_row)  
            progress_bar(idx+1, len(self.tracks))
        print('\n') 
        genre_latents = pd.DataFrame(genre_rows)
        genre_latents = genre_latents.groupby('genre').mean().dropna()
        genre_latents = genre_latents.reset_index()
        print(f'Genre distributions built. {round((time.time()-start_time)/60,2)} minutes elapsed')
        
        self.genres = genre_latents

        self.tracks = self.tracks.drop(columns=['artist_genres'])
        
        print('Latent Space Built.')

        #fit umap transformer

    def save_db(self, save_folder):

        create_new_directories(save_folder)

        #save tracks

        feather.write_feather(self.tracks, save_folder+'/tracks.feather')

        #save genre space

        feather.write_feather(self.genres, save_folder+'/genres.feather')

        #save scaler

        dump(self._scaler, save_folder+'/std_scaler.bin', compress=True)


        #save umap transformer
    
    def load_db(self, save_folder):
        try:
            self.tracks = feather.read_feather(save_folder+'/tracks.feather')
            print('Loaded tracks.')
        except:
            print('Failed to load tracks.')

        
        try:
            self.genres = feather.read_feather(save_folder+'/genres.feather')
            print('Loaded genres.')
        except:
            print('Failed to load genres.')

        try:
            self._scaler=load(save_folder+'/std_scaler.bin')
            print('loaded scaler')
        except:
            print('Failed to load scaler.')
    
    def train(self, mel_directory, epochs, train_test_split=0.2, sample_size=None, batch_size=32):

        from src.AudioDataGenerator import AudioDataGenerator

        #check for loaded/built model

        #add later

        #create train test generator
        input_size = self._get_input_shape(mel_directory)

        self.train_test_generator = AudioDataGenerator(batch_size=batch_size, input_size=input_size, output_size=(self.image_height, self.image_width), directory=mel_directory, shuffle=True, train_test_split=True, test_size=train_test_split, sample_size=sample_size, shorten_factor=self.final_shorten_factor, mel_gamma=self.mel_gamma)

        #train
        self.history_ = self.autoencoder.fit(self.train_test_generator.train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=self.train_test_generator.test)


    def _get_input_shape(self, file_directory):
        test_files = list(map(lambda x: file_directory + '/' + x, os.listdir(file_directory)[:5]))
        test_arr = []
        for file in test_files:
            test_arr.append(np.load(file).shape)
        test_arr = np.array(test_arr)
        input_shape = (test_arr[:,0].max(),test_arr[:,1].max())

        return input_shape
    
    def search_for_recommendations(self, query, num=10, popularity_threshold=10, get_time_and_freq=False, save_folder='data'):
        
        from src.AppAudioDataGenerator import AppAudioDataGenerator

        create_new_directories(save_folder)

        #get query result
        id_ = self._spotify.search(query, type='track')['tracks']['items'][0]['id']
        track = self._spotify.track(id_)
        link = track['preview_url']

        if link is not None:
            #convert query to mel
            self.asc = AudioSpectrogramConverter(save_folder=save_folder, 
                                        fft_size=self.fft_size,
                                        spec_thresh=self.spec_thresh,
                                        n_mel_freq_components=self.n_mel_freq_components,
                                        shorten_factor=self.shorten_factor,
                                        start_freq=self.start_freq,
                                        end_freq=self.end_freq)

            mel = self.asc.convert(link=link, file_id=track['id'])

            prediction_gen = AppAudioDataGenerator(batch_size=1, input_size=(mel.shape[0], mel.shape[0]), output_size=(self.image_height, self.image_width), shorten_factor=self.shorten_factor, mel_gamma=self.mel_gamma)

            mel_batch = prediction_gen.get_tensors_from_data(data=mel, num_tiles=self.num_tiles)
            # mel_batch = prediction_gen.get_tensors_from_data(data=mel, num_tiles=None)


            mel_batch = np.array(mel_batch)

            mel_batch = self._scaler.transform([self.run_inference(mel_batch[0]).mean(axis=0)])
            # mel_batch = self._scaler.transform(self.run_inference(mel_batch[0]))

            #get model prediciton of query
            vector = pd.DataFrame(mel_batch, columns=self.latent_cols)

            #get similarity
            similarity = self.get_similarity(vector, self.tracks, subset=self.latent_cols, num=50, popularity_threshold=popularity_threshold)

            similarity = similarity[~similarity.track_name.apply(lambda x: x.lower()).isin([track['name'].lower()])][:num].reset_index()

            #return track, recommendations latent space/similarities
            if get_time_and_freq:
                similarity['time_similarity'] = self.get_similarity(vector, similarity, subset=self.latent_cols[:len(self.latent_cols)//2], num=num, popularity_threshold=popularity_threshold, sort_tracks=False)['similarity']
                similarity['frequency_similarity'] = self.get_similarity(vector, similarity, subset=self.latent_cols[len(self.latent_cols)//2:], num=num, popularity_threshold=popularity_threshold, sort_tracks=False)['similarity']

                return track, similarity[['track_name','track_uri','artist_name','similarity','track_popularity','time_similarity','frequency_similarity']], similarity, vector
            else:
                return track, similarity[['track_name','track_uri','artist_name','similarity','track_popularity']], similarity, vector

        else:
            print('No Preview Available. Try a different search.')
    
    def run_inference(self, X):

        # batch == 1
        # input shape == (1, height, width, 1)

        X = X.astype(np.float32)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        input_shape = input_details[0]['shape']

        self.interpreter.resize_tensor_input(input_details[0]['index'], (X.shape[0], input_shape[1], input_shape[2], input_shape[3]))
        self.interpreter.allocate_tensors()

        # pass X through encoder
        self.interpreter.set_tensor(input_details[0]['index'], X)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        return output_data

    def get_similarity(self, df1, df2, subset, num=10, similarity_measure='cosine', popularity_threshold=0, sort_tracks=True):
        if similarity_measure == 'cosine':
            similarity_measure_fn = cosine_similarity
            sort = False
        elif similarity_measure == 'euclidean':
            similarity_measure_fn = euclidean_distances
            sort = True
        else:
            raise ValueError('similarity_measure must be "cosine" or "euclidean"')

        similarity = similarity_measure_fn(np.array(df1[subset]), np.array(df2[subset]))

        similarity_df = df2.copy()
        similarity_df['similarity'] = similarity.T

        if popularity_threshold > 0:
            similarity_df = similarity_df[similarity_df.track_popularity > popularity_threshold]

        if sort_tracks:
            similarity_df = similarity_df.sort_values(by='similarity', ascending=sort).reset_index()

        return similarity_df[:num]

    def plot_genre_space(self, track, recommendations):
        pass

def create_new_directories(save_folder):

    directories = save_folder.split('/')
    save_folders=[directories[0]]
    for directory in directories[1:]:
        save_folders.append(save_folders[-1] + '/' + directory)
    for folder in save_folders:
        try:
            os.mkdir(folder)
        except:
            pass

def progress_bar(progress, total, display_length=60):
        left_ratio = display_length * progress//total
        right_ratio = display_length - left_ratio
        
        print('['+ '='*left_ratio + '>' + '.'*right_ratio + f'] {progress} / {total}', end='\r') 
