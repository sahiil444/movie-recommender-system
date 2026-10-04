from __future__ import annotations
from pathlib import Path
import pandas as pd

MOVIE_COLUMNS = ["id", "title", "original_title", "overview", "genres", "keywords"]

DEFAULT_MOVIES_FILENAME = "tmdb_5000_movies.csv"
DEFAULT_CREDITS_FILENAME = "tmdb_5000_credits.csv"


def get_raw_data_dir() -> Path:

    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "raw"


def load_raw_data(raw_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:

    raw_dir = Path(raw_dir) if raw_dir is not None else get_raw_data_dir()

    movies_path = raw_dir / DEFAULT_MOVIES_FILENAME
    credits_path = raw_dir / DEFAULT_CREDITS_FILENAME

    for path in (movies_path, credits_path):
        if not path.is_file():
            raise FileNotFoundError(f"Raw data file not found: {path}")

    df_movies = pd.read_csv(movies_path)
    df_credits = pd.read_csv(credits_path)
    return df_movies, df_credits


def merge_datasets(df_movies: pd.DataFrame, df_credits: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MOVIE_COLUMNS if c not in df_movies.columns]
    if missing:
        raise KeyError(f"Missing expected movies columns: {missing}")
    if "movie_id" not in df_credits.columns:
        raise KeyError("Missing expected credits column: 'movie_id'")
    if "title" not in df_credits.columns:
        raise KeyError("Missing expected credits column: 'title'")

    movies = df_movies[MOVIE_COLUMNS]
    credits = df_credits.drop(columns="title")
    movies = movies.merge(credits, left_on="id", right_on="movie_id")
    movies = movies.drop(columns=["id", "original_title"])
    return movies


def run_data_ingestion(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the raw TMDB files and return the merged ingestion DataFrame.

    Args:
        raw_dir: Optional override for the raw data directory.

    Returns:
        The merged DataFrame of movies and credits.
    """
    df_movies, df_credits = load_raw_data(raw_dir)
    return merge_datasets(df_movies, df_credits)
