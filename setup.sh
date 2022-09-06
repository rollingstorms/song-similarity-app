mkdir -p ~/.streamlit/
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

echo "\
clientId = '5f818fbc1c374e239ee0139163734165'\n\
clientSecret = 'dd3e1522369f4413bfadc8613cf7bf43'\n\
" > ~/.streamlit/secrets.toml