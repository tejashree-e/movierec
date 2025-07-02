import streamlit as st
import requests
import pandas as pd
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity
import os

genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "ml-100k")
    ratings_path = os.path.join(data_dir, "u.data")
    movies_path = os.path.join(data_dir, "u.item")

    ratings = pd.read_csv(ratings_path, sep="\t", names=['user_id','movie_id','rating','timestamp'])
    movies = pd.read_csv(movies_path, sep="|", encoding='latin-1', names=[
        'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    return ratings, movies

@st.cache_data
def prepare_data(ratings, movies):
    data = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')
    user_matrix = data.pivot_table(index='user_id', columns='title', values='rating')
    movie_user_matrix = user_matrix.T.fillna(0)
    similarity = cosine_similarity(movie_user_matrix)
    similarity_df = pd.DataFrame(similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)
    return user_matrix, similarity_df

def fetch_movie_info(title, api_key):
    base_title = title.split(" (")[0]
    encoded_title = quote(base_title)
    url = f"http://www.omdbapi.com/?t={encoded_title}&apikey={api_key}"
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

api_key = 'eac0b877'

ratings, movies = load_data()
user_matrix, similarity_df = prepare_data(ratings, movies)

st.subheader("Rate some movies you like:")
movies_to_rate = st.multiselect("Select movies to rate", options=user_matrix.columns, max_selections=5)

user_ratings = {}
for movie in movies_to_rate:
    rating = st.slider(f"Rate '{movie}'", 1, 5, 3)
    user_ratings[movie] = rating

if user_ratings:
    st.write("Thanks for rating!")

st.title("Movie Recommender")

selected_genres = st.multiselect("Filter by genre(s):", options=genre_columns)

filtered_movies = movies
if selected_genres:
    genre_filter = movies[selected_genres].sum(axis=1) > 0
    filtered_movies = movies[genre_filter]

filtered_titles = filtered_movies['title'].values.tolist()

search_term = st.text_input("Search for a movie:")

if search_term:
    filtered_titles = [m for m in filtered_titles if search_term.lower() in m.lower()]

if filtered_titles:
    movie = st.selectbox("Select a Movie:", options=filtered_titles)
else:
    st.write("No movies found matching your criteria.")
    movie = None

if movie:
    st.write(f"Top 10 movies similar to **{movie}**:")
    sim_scores = similarity_df[movie].sort_values(ascending=False)[1:11]
    for i, (title, score) in enumerate(sim_scores.items(), 1):
        st.markdown(f"### {i}. {title} ({score:.3f})")

        info = fetch_movie_info(title, api_key)
        if info:
            poster_url = info.get('poster')
            if poster_url and poster_url != "N/A":
                poster_url = poster_url.replace("http://", "https://")
                st.image(poster_url, width=150)
            else:
                st.write("_Poster not available_")

            if info.get('plot'):
                st.write(f"**Plot:** {info['plot']}")
            if info.get('year'):
                st.write(f"**Year:** {info['year']}")
        else:
            st.write("_Details not available or API limit reached._")

        st.markdown("---")
