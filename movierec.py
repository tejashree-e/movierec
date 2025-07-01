import streamlit as st
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load Data ----
@st.cache_data
def load_data():
    ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=['user_id','movie_id','rating','timestamp'])
    movies = pd.read_csv("ml-100k/u.item", sep="|", encoding='latin-1', names=[
        'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    return ratings, movies

# ---- Prepare Matrix and Similarity ----
@st.cache_data
def prepare_data(ratings, movies):
    data = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')
    user_matrix = data.pivot_table(index='user_id', columns='title', values='rating')
    movie_user_matrix = user_matrix.T.fillna(0)
    similarity = cosine_similarity(movie_user_matrix)
    similarity_df = pd.DataFrame(similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)
    return user_matrix, similarity_df

# ---- Fetch Poster & Info from OMDb ----
@st.cache_data(show_spinner=False)
def fetch_movie_info(title, api_key):
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    if data.get('Response') == 'False':
        return None
    return {
        'poster': data.get('Poster'),
        'year': data.get('Year'),
        'plot': data.get('Plot')
    }

# ---- API Key ----
api_key = 'e050f40c'  # Replace with your real OMDb API key

# ---- Streamlit App ----
st.title("🎬 Movie Recommender")

ratings, movies = load_data()
user_matrix, similarity_df = prepare_data(ratings, movies)

movie = st.selectbox("Select a Movie:", options=user_matrix.columns)

if movie:
    st.write(f"Top 10 movies similar to **{movie}**:")
    sim_scores = similarity_df[movie].sort_values(ascending=False)[1:11]

    for i, (title, score) in enumerate(sim_scores.items(), 1):
        st.markdown(f"### {i}. {title} ({score:.3f})")
        clean_title=title.split("(")[0]
        info = fetch_movie_info(title, api_key)
        if info:
            if info['poster'] and info['poster'] != "N/A":
                st.image(info['poster'], width=150)
            if info['plot']:
                st.write(f"**Plot:** {info['plot']}")
            if info['year']:
                st.write(f"**Year:** {info['year']}")
        else:
            st.write("_Details not available._")

        st.markdown("---")
