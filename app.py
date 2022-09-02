import streamlit as st
import streamlit.components.v1 as components
from src.LatentSpace import LatentSpaceApp
import umap
from joblib import load
import pandas as pd
import plotly.express as px


def search(query):
	st.experimental_set_query_params(query=query)
	with st.spinner('Loading Tracks...'):
		autoencoder_path = 'data/encoder'
		latent_space = LatentSpaceApp(latent_dims=256, num_tiles=64)
		latent_space.load(autoencoder_path)
		tab1, tab2 = st.tabs(['Song Recommendations', 'Latent Space Visualization'])
		# try:
		with tab1:
			track, df, latents, this_track= latent_space.search_for_recommendations(query, get_time_and_freq=True)
			display_song('This Song', track_id=track['id'], track_name=track['name'], artist=track["artists"][0]["name"])

			for idx, row in df.iterrows():
				track_id = row.track_uri.split(':')[-1]
				display_song(idx+1, 
							track_id=track_id,
							track_name=row.track_name,
							artist=row.artist_name,
							similarity=row.similarity,
							time_similarity=row.time_similarity,
							freq_similarity=row.frequency_similarity)

		with tab2:
			fig = plot_genre_space(track, this_track, latents, latent_space)
			st.header('Genres in the Sonic Landscape')
			st.subheader(track['name'] + ' by ' + track['artists'][0]['name'])
			st.plotly_chart(fig, use_container_width=True)
			st.write('Using dimensionality reduction, basic genres and similar songs can be plotted in a visualization of the latent space that is created by an encoder.')


		# except:
		# 	st.write("""No preview for this song or artist on Spotify. Please try a different search.
		# 		This recommendation system relies on the preview mp3's provided by Spotify's Public API.
		# 		If there is no mp3 preview, Sonufy can't make a recommendation.""")

def display_song(index, track_id, track_name, artist, similarity=None, time_similarity=None, freq_similarity=None):
	col1, col2, col3 = st.columns((1,3,3))
	if str(index).isnumeric():
		if index % 2 == 1:
			middle = col2
			end = col3
		else:
			middle = col3
			end = col2
	else:
		middle = col3
		end = col2

	with col1:
		st.header(index)
	with middle:
		st.subheader(f'{track_name} by {artist}')
		if similarity != None:
			st.write(f'Similarity: {round(similarity,2)}')
		if time_similarity != None:
			st.write(f'Time Similarity: {round(time_similarity,2)}')
		if freq_similarity != None:
			st.write(f'Frequency Similarity: {round(freq_similarity,2)}')
	with end:
		components.iframe(f'https://open.spotify.com/embed/track/{track_id}', width=250, height=250)

def plot_genre_space(track, this_track, latents, latent_space):
	this_track_df = pd.DataFrame(this_track, columns=latent_space.latent_cols)
	this_track_df['name'] = track['name'] + ' - ' + track['artists'][0]['name']
	this_track_df['label'] = 2

	latents['name'] = latents['track_name'] + ' - ' + latents['artist_name']
	latents['label'] = 1
	latents = latents[['name'] + latent_space.latent_cols + ['label']]
	
	autoencoder_path = 'data/autoencoder_256dim_time_freq_128k_20'

	genres_and_tracks = pd.concat([latent_space.genres, latents, this_track_df]).reset_index(drop=True)
	genre_map = load(autoencoder_path+ '/genre_map.bin')
	genre_map_trans = genre_map.transform(genres_and_tracks[latent_space.latent_cols])

	genre_map_df = pd.DataFrame(genre_map_trans, columns=['x','y'])
	genre_map_df = pd.concat([genres_and_tracks[['name','label']], genre_map_df], axis=1)
	genre_map_df.label = genre_map_df.label.map({0:'genre', 1:'similar song', 2:'this song'})
	genre_map_df['annotation'] = genre_map_df.apply(lambda x: x['name'] if x['label'] == 'genre' else '', axis=1)

	fig = px.scatter(genre_map_df, x='x', y='y', color='label', hover_name='name', size=[.5]*len(genre_map_df), width=800, height=600, text='annotation')
	return fig		


try:
	params = st.experimental_get_query_params()
	params_query = params['query'][0]
except:
	params_query = ''
st.header('Sonufy')
query = st.text_input('search for a track on Spotify to hear similar songs:', value='')
if query != '':
	search(query)
elif params_query != '':
	search(params_query)
