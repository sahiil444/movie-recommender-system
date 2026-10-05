# Movie Recommender System

A content-based movie recommender built with Streamlit. Pick a movie and the app
recommends 6 similar movies, with posters and details fetched from the TMDB API.

## Features

- Searchable movie dropdown (4800 movies)
- Top-6 recommendations based on movie metadata
- Movie cards with poster, title, release year and rating
- Details page for each recommendation (overview, genres, runtime, TMDB link)
- Poster fetching with fallback and retries via the TMDB API

## How It Works

The system is content-based. For every movie, a combined `tags` text is built
from its overview, genres, keywords, cast and crew. These tags are converted
into TF-IDF vectors, and a cosine similarity matrix between all movies is
precomputed. At prediction time, the selected movie's most similar movies are
returned as recommendations.

## Tech Stack

- Python 3.14, managed with `uv`
- pandas, NumPy, scikit-learn, NLTK
- Streamlit
- Requests + TMDB API

## Project Structure

```
app.py                      # Streamlit frontend
artifacts/                  # Saved pickle artifacts (df, tfidf, similarity matrix)
data/raw/                   # TMDB 5000 movies + credits CSVs
notebook/                   # Exploration and model comparison
src/movie_recommender_system/
    components/             # ingestion, preprocessing, model training
    pipeline/               # training and prediction pipelines
    tmdb_api.py             # TMDB API helper (token, retries, poster fallback)
```

## Setup & Run

1. Install dependencies:

```bash
uv sync
```

2. Add your TMDB Read Access Token to `.streamlit/secrets.toml`:

```toml
TMDB_API_READ_ACCESS_TOKEN = "your_token_here"
```

3. The required pickle artifacts (`processed_df.pkl`, `tfidf.pkl`,
`similarity_matrix.pkl`) must already exist in `artifacts/`. If they are
missing, regenerate them by running the training pipeline. Note: NLTK corpora
(`stopwords`, `wordnet`, `punkt`, `averaged_perceptron_tagger`) must be
downloaded first.

4. Run the app from the project root:

```bash
streamlit run app.py
```

## Limitations

- Recommendations are purely content-based; there is no user personalization.
- Posters and details depend on the live TMDB API (network issues can delay or
  block them).
- Movie matching relies on exact titles, so duplicate or ambiguous titles may
  not always resolve to the intended movie.
