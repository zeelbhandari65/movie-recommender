import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    return df

df = load_data()

# Preprocessing: Fill NA
df['overview'] = df['overview'].fillna("")

# TF-IDF Vectorizer on 'overview'
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create reverse mapping of indices and titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation Function
def recommend(title, num=5):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("This app recommends movies based on content similarity from the movie's description.")

movie_list = df['title'].dropna().unique()
selected_movie = st.selectbox("Select a movie:", sorted(movie_list))

if st.button("Get Recommendations"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.success("Top Recommended Movies:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("No recommendations found. Try another movie.")

with st.expander("📄 Raw Data"):
    st.dataframe(df[['title', 'overview']].head(10))

with st.expander("ℹ️ About"):
    st.markdown("""
    - This app uses **TF-IDF vectorization** on movie overviews.
    - Then computes **cosine similarity** to recommend similar movies.
    - Dataset source: `movies.csv`
    """)
