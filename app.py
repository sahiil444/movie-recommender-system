from __future__ import annotations

import streamlit as st

from src.movie_recommender_system.pipeline.prediction_pipeline import (
    PredictionPipeline,
)
from src.movie_recommender_system.tmdb_api import (
    TMDBAPIError,
    fetch_movie_details,
    fetch_poster_bytes,
)

# ---------------------------------------------------------------------------
# Page + theme setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #f5f5f5; }
    .movie-card-title { font-weight: 600; margin-top: 0.4rem; font-size: 0.85rem; }
    .movie-card-meta { color: #b0b0b0; font-size: 0.75rem; }
    .poster-placeholder {
        background-color: #1f2630;
        border-radius: 8px;
        width: 180px;
        max-width: 100%;
        aspect-ratio: 2 / 3;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6c7686;
        font-size: 2rem;
    }
    .selected-card {
        background-color: #161b22;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    div[data-testid="stImage"] img {
        border-radius: 8px;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached backend loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_prediction_pipeline() -> PredictionPipeline:
    """Create and cache the PredictionPipeline so it loads only once."""
    return PredictionPipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_movie_title(title: str) -> str:
    """Display-only cleanup: strip brackets and leading '#' from titles."""
    cleaned = title.replace("[", "").replace("]", "").replace("#", "")
    return cleaned.strip()


def sort_key(title: str) -> tuple[int, str]:
    """Alphabetic titles first; titles starting with digits/special chars later."""
    first = title.lstrip()[:1]
    normal_first = 0 if first.isalpha() else 1
    return (normal_first, title.lower())


def get_movie_id_for_title(pipeline: PredictionPipeline, title: str):
    """Map a recommended title back to its TMDB movie_id in processed_df."""
    match = pipeline.processed_df[pipeline.processed_df["title"] == title]
    if match.empty:
        return None
    return int(match.iloc[0]["movie_id"])


POSTER_CARD_WIDTH = 180   # px, compact poster thumbnail
POSTER_DETAILS_WIDTH = 300  # px, larger poster on the details page


def render_poster(details, width: int = POSTER_CARD_WIDTH) -> None:
    poster_bytes = None
    if details and details.get("poster_url"):
        try:
            poster_bytes = fetch_poster_bytes(details["poster_url"])
        except TMDBAPIError:
            poster_bytes = None

    if poster_bytes:
        st.image(poster_bytes, width=width)
    else:
        st.markdown(
            f'<div class="poster-placeholder" style="width:{width}px">🎬</div>',
            unsafe_allow_html=True,
        )


def render_movie_card(rank: int, title: str, details) -> None:
    render_poster(details)
    st.markdown(
        f'<div class="movie-card-title">{rank}. {format_movie_title(title)}</div>',
        unsafe_allow_html=True,
    )
    if details:
        year = details["release_year"]
        rating = details["rating"]
        rating_text = f"⭐ {rating:.1f}" if isinstance(rating, (int, float)) else "⭐ N/A"
        st.markdown(
            f'<div class="movie-card-meta">{year} &nbsp;•&nbsp; {rating_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="movie-card-meta">Details unavailable</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def render_details_view(movie_id: int) -> None:
    """Movie details page for a single TMDB movie."""
    if st.button("← Back"):
        st.session_state.pop("selected_tmdb_id", None)
        st.rerun()

    with st.spinner("Loading movie details..."):
        try:
            details = fetch_movie_details(movie_id)
        except TMDBAPIError as exc:
            st.error(str(exc))
            return

    if details is None:
        st.warning("This movie could not be found on TMDB.")
        return

    col_poster, col_info = st.columns([1, 2])

    with col_poster:
        render_poster(details, width=POSTER_DETAILS_WIDTH)

    with col_info:
        st.header(details["title"])
        rating = details["rating"]
        rating_text = f"⭐ {rating:.1f}/10" if isinstance(rating, (int, float)) else "⭐ N/A"
        runtime = details["runtime"]
        runtime_text = f"{runtime} min" if runtime else "N/A"
        st.write(f"**Release year:** {details['release_year']}")
        st.write(f"**Rating:** {rating_text}")
        st.write(f"**Genres:** {', '.join(details['genres']) if details['genres'] else 'N/A'}")
        st.write(f"**Runtime:** {runtime_text}")
        st.write(f"**Overview:** {details['overview']}")
        st.link_button("View on TMDB ↗", details["tmdb_url"])


def render_recommendation_view(pipeline: PredictionPipeline) -> None:
    st.title("🎬 Movie Recommender System")
    st.write(
        "Select a movie you like, and we'll recommend 6 similar movies "
        "based on its content."
    )

    movie_titles = sorted(pipeline.processed_df["title"].tolist(), key=sort_key)

    col_search, col_button = st.columns([5, 1], gap="small", vertical_alignment="bottom")
    with col_search:
        selected_movie = st.selectbox(
            "Choose a movie:",
            movie_titles,
            index=None,
            placeholder="Search or select a movie...",
            format_func=format_movie_title,
        )
    with col_button:
        recommend_clicked = st.button("Recommend", type="primary")

    if selected_movie:
        selected_id = get_movie_id_for_title(pipeline, selected_movie)
        selected_details = None
        if selected_id is not None:
            try:
                selected_details = fetch_movie_details(selected_id)
            except TMDBAPIError:
                selected_details = None

        with st.container(border=True):
            poster_col, info_col = st.columns([1, 5], gap="medium", vertical_alignment="center")
            with poster_col:
                render_poster(selected_details, width=100)
            with info_col:
                title = selected_details["title"] if selected_details else format_movie_title(selected_movie)
                st.markdown("**You picked**")
                st.subheader(title)
                if selected_details:
                    year = selected_details["release_year"]
                    rating = selected_details["rating"]
                    rating_text = f"{rating:.1f}" if isinstance(rating, (int, float)) else "N/A"
                    genres = ", ".join(selected_details["genres"][:3]) if selected_details["genres"] else ""
                    meta = f"{year} · {rating_text}"
                    if genres:
                        meta += f" · {genres}"
                    st.markdown(meta)

    if recommend_clicked:
        if not selected_movie:
            st.warning("Please select a movie first.")
            return

        with st.spinner("Finding similar movies..."):
            try:
                recommendations = pipeline.predict(selected_movie)
            except Exception:  # noqa: BLE001 - friendly message only
                st.error("Could not generate recommendations for that movie.")
                return

        st.session_state["recommendations"] = recommendations
        st.session_state["source_movie"] = selected_movie

    recommendations = st.session_state.get("recommendations")
    if not recommendations:
        return

    source = st.session_state.get("source_movie", "your movie")
    st.subheader(f"Because you liked **{format_movie_title(source)}**:")

    cols_per_row = 6
    row = recommendations[:cols_per_row]
    cols = st.columns(cols_per_row, gap="small")
    for j, title in enumerate(row):
        i = j
        movie_id = get_movie_id_for_title(pipeline, title)
        details = None
        if movie_id is not None:
            try:
                details = fetch_movie_details(movie_id)
            except TMDBAPIError:
                details = None

        with cols[j]:
            render_movie_card(i + 1, title, details)
            if movie_id is not None and st.button("View", key=f"view_{i}_{movie_id}"):
                st.session_state["selected_tmdb_id"] = movie_id
                st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline = load_prediction_pipeline()

    selected_id = st.session_state.get("selected_tmdb_id")
    if selected_id is not None:
        render_details_view(selected_id)
    else:
        render_recommendation_view(pipeline)


if __name__ == "__main__":
    main()
