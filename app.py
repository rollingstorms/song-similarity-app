import streamlit as st
import streamlit.components.v1 as components
from src.Sonufy import Sonufy
import pandas as pd
import plotly.express as px
import time
from copy import deepcopy

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_sonufy():
	model_path = 'test_64dim'
	sonufy = Sonufy(latent_dims=64, output_size=(64,64))
	sonufy.load_encoder(model_path, app=True)
	sonufy.load_db(model_path)

	return sonufy



def search(query):
	st.experimental_set_query_params(query=query)
	status = st.empty()
	with status.container():
		progress_bar = st.progress(0)
		st.write('Loading Tracks...')

	sonufy = load_sonufy()
	progress_bar.progress(10)

	tab1, tab2 = st.tabs(['Song Recommendations', 'Latent Space Visualization'])
	try:
		with tab1:
			track, df, latents, this_track= sonufy.search_for_recommendations(query, get_time_and_freq=True)
			progress_bar.progress(50)
			display_song('This Song', track_id=track['id'], track_name=track['name'], artist=track["artists"][0]["name"])
			progress_bar.progress(75)
			for idx, row in df.iterrows():
				track_id = row.track_uri.split(':')[-1]
				display_song(idx+1, 
							track_id=track_id,
							track_name=row.track_name,
							artist=row.artist_name,
							similarity=row.similarity,
							time_similarity=row.time_similarity,
							freq_similarity=row.frequency_similarity)
			progress_bar.progress(100)

		with tab2:
			# fig = plot_genre_space(track, this_track, latents, latent_space)
			# st.header('Genres in the Sonic Landscape')
			# st.subheader(track['name'] + ' by ' + track['artists'][0]['name'])
			# st.plotly_chart(fig, use_container_width=True)
			st.write('Using dimensionality reduction, basic genres and similar songs can be plotted in a visualization of the latent space that is created by an encoder.')

		time.sleep(.5)
		status.empty()

	except:
		st.warning("""No preview for this song or artist on Spotify. Please try a different search.
			This recommendation system relies on the preview mp3's provided by Spotify's Public API.
			If there is no mp3 preview, Sonufy can't make a recommendation.""")

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

try:
	params = st.experimental_get_query_params()
	params_query = params['query'][0]
except:
	params_query = ''
# st.header('Sonufy')
st.image('img/Sonufy_logo.png', width=400)
query = st.text_input('search for a track on Spotify to hear similar songs:', value='')
if query != '':
	search(query)
elif params_query != '':
	search(params_query)
