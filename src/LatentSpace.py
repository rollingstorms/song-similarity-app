import tensorflow as tf
import numpy as np
import pandas as pd
from pyarrow import feather
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from src.DataGenerator import AppDataGenerator
from joblib import load
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import os
from copy import deepcopy

class LatentSpaceApp:
    
    def __init__(self,
                latent_dims=256,
                output_size=(128,128),
                num_tiles=64):

        self.prediction_generator = AppDataGenerator(
                                    image_size=(128,512),
                                    output_size=output_size)

        self.latent_cols = [f'latent_{i}' for i in range(latent_dims)]
        self._num_tiles = num_tiles

        # f = open('data/apikeys/.apikeys3.json')
        # apikeys = json.load(f)
        # client_id = apikeys['clientId']
        # client_secret = apikeys['clientSecret']
        client_id = st.secrets['clientId']
        client_secret = st.secrets['clientSecret']

        credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

        self._spotify = spotipy.Spotify(client_credentials_manager=credentials_manager)

    @st.cache(show_spinner=False, allow_output_mutation=True)
    def _load_autoencoder(self, directory_to_load):
        return tf.keras.models.load_model(directory_to_load+'/encoder.h5')

    # @st.cache(show_spinner=False)
    def _load_feather(self, directory_to_load, file):
        return feather.read_feather(directory_to_load+'/'+file)
        
    def load(self, directory_to_load):
        try:
            start = time.time()
            self.encoder = deepcopy(self._load_autoencoder(directory_to_load))
            print('Loaded encoder.')
            st.write(f'encoder {round(time.time() - start, 2)}')
        except:
            print('Failed to load encoder.')

        try:
            start = time.time()
            self.tracks = deepcopy(self._load_feather(directory_to_load, 'tracks.feather'))
            print('Loaded tracks.')
            st.write(f'tracks {round(time.time() - start, 2)}')

        except:
            print('Failed to load tracks.')

        try:
            self.genres = self._load_feather(directory_to_load, 'genres.feather')
            print('Loaded genres.')
            st.write('genres')

        except:
            print('Failed to load genres.')

        try:
            self._scaler = load(directory_to_load+'/std_scaler.bin')
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
        vector = np.array(self.encoder(img[0])).mean(axis=0)
        vector = self._scaler.transform(pd.DataFrame([vector], columns=self.latent_cols))
        vector = pd.DataFrame(vector, columns=self.latent_cols)
        return vector

    def search_for_recommendations(self, query, num=10, popularity_threshold=10, get_time_and_freq=False):
        id_ = self._spotify.search(query, type='track')['tracks']['items'][0]['id']
        track = self._spotify.track(id_)
        link = track['preview_url']
        st.write(track['name'])
        st.write(track['artists'][0]['name'])
        st.write(link)

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