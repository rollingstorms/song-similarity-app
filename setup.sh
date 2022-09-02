mkdir -p ~/.streamlit/
mkdir -p ~/data/Spotify/mp3s
echo "\
[general]\n\
email = \"mikerothart@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml