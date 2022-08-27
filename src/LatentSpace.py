import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyarrow import feather
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from src.DataGenerator import AudioDataGenerator
from src.helper_functions import progress_bar
from joblib import dump, load
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import os


class LatentSpaceApp:
    
    def __init__(self,
                autoencoder_path,
                image_dir='data/Spotify/comp_pngs',
                sample_size=None,
                latent_dims=128,
                num_channels=1,
                output_size=(128,128),
                scale=True,
                threshold_level=0,
                num_tiles=4):
        self._batch_size = 1
        self.autoencoder = tf.keras.models.load_model(autoencoder_path)
        self.prediction_generator = AudioDataGenerator(directory=image_dir,
                                    image_size=(128,512),
                                    color_mode='rgb',
                                    batch_size=1, 
                                    shuffle=False,
                                    sample_size = sample_size,
                                    output_channel_index=0,
                                    num_output_channels=num_channels,
                                    output_size=output_size,
                                    threshold_level=threshold_level)
        self.latent_cols = [f'latent_{i}' for i in range(latent_dims)]
        
        self.size = self.prediction_generator.size
        self._num_channels = num_channels
        self._scale = scale
        self._num_tiles = num_tiles

        # f = open('data/apikeys/.apikeys.json')
        # apikeys = json.load(f)
        client_id = st.secrets['clientId']
        client_secret = st.secrets['clientSecret']

        credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

        self._spotify = spotipy.Spotify(client_credentials_manager=credentials_manager)
        
    
    def load(self, directory_to_load, load_full_results=False):
        try:
            tracks_folder = directory_to_load + '/tracks'
            tracks_folder_list = os.listdir(tracks_folder)
            self.tracks = pd.DataFrame()
            for track in tracks_folder_list:
                self.tracks = pd.concat([self.tracks, feather.read_feather(tracks_folder + "/" + track)])
            self.tracks = self.tracks.sort_index()
            print('Loaded tracks.')
        except:
            print('Failed to load tracks.')

        try:
            self.genres = feather.read_feather(directory_to_load+'/genres.feather')
            print('Loaded genres.')
        except:
            print('Failed to load genres.')

        try:
            self._scaler=load(directory_to_load+'/std_scaler.bin')
            print('loaded scaler')
        except:
            print('Failed to load scaler.')

        try:
            self._genre_map = load(directory_to_load+'/genre_map.bin')
            print('loaded genre map')
        except:
            print('Failed to load genre map')

        

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


    def get_vector_from_preview_link(self, link, track_id):
        img = self.prediction_generator.get_vector_from_preview_link(link, track_id, num_tiles=self._num_tiles)
        vector = np.array(self.autoencoder.encoder(img[0])).mean(axis=0)
        vector = self._scaler.transform(pd.DataFrame([vector], columns=self.latent_cols))
        vector = pd.DataFrame(vector, columns=self.latent_cols)
        return vector

    def search_for_recommendations(self, query, num=10, popularity_threshold=10, get_time_and_freq=False):
        id_ = self._spotify.search(query, type='track')['tracks']['items'][0]['id']
        track = self._spotify.track(id_)
        link = track['preview_url']
        print(track['name'])
        print(track['artists'][0]['name'])
        print(link)

        if link is not None:

            vector = self.get_vector_from_preview_link(link, id_)
            similarity = self.get_similarity(vector, self.tracks, subset=self.latent_cols, num=50, popularity_threshold=popularity_threshold)

            similarity = similarity[~similarity.track_name.apply(lambda x: x.lower()).isin([track['name'].lower()])][:num].reset_index()

            if get_time_and_freq:
                similarity['time_similarity'] = self.get_similarity(vector, similarity, subset=self.latent_cols[:len(self.latent_cols)//2], num=num, popularity_threshold=popularity_threshold, sort_tracks=False)['similarity']
                similarity['frequency_similarity'] = self.get_similarity(vector, similarity, subset=self.latent_cols[len(self.latent_cols)//2:], num=num, popularity_threshold=popularity_threshold, sort_tracks=False)['similarity']
            
                return track, similarity[['track_name','track_uri','artist_name','similarity','track_popularity','time_similarity','frequency_similarity']], similarity, vector
            else:
                return track, similarity[['track_name','track_uri','artist_name','similarity','track_popularity']], similarity, vector
        else:
            print('No Preview Available. Try a different search.')