
from __future__ import annotations
import os
import time
from typing import Any, Optional

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0


def _build_session() -> requests.Session:
    """Create a shared HTTP session with automatic status-code retries."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _build_session()


class TMDBAPIError(Exception):
    """Raised when a TMDB request fails in a user-friendly way."""


def get_api_read_access_token() -> Optional[str]:
    """Return the TMDB Read Access Token from secrets or the environment."""
    try:
        token = st.secrets.get("TMDB_API_READ_ACCESS_TOKEN", "")
        if token:
            return token
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return os.environ.get("TMDB_API_READ_ACCESS_TOKEN") or None


def _get_json(url: str, headers: dict[str, str], params: Optional[dict] = None) -> Optional[dict]:
    """GET a TMDB JSON endpoint with retries.

    Returns the parsed JSON dict, or ``None`` for a genuine 404.
    Raises TMDBAPIError on auth failure, server errors, and network errors
    after all retries so that failures are never cached.
    """
    last_error: Optional[Exception] = None
    response: Optional[requests.Response] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _SESSION.get(
                url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if response is None:
        if isinstance(last_error, requests.exceptions.Timeout):
            raise TMDBAPIError("TMDB request timed out. Please try again later.") from None
        raise TMDBAPIError("Could not reach TMDB. Please try again in a moment.") from None

    if response.status_code == 404:
        return None  # permanent: not found — do not retry
    if response.status_code in (401, 403):
        raise TMDBAPIError("TMDB rejected the API token. Please check your credentials.")
    if response.status_code != 200:
        raise TMDBAPIError(f"TMDB request failed (status {response.status_code}).")

    return response.json()


def _auth_headers() -> dict[str, str]:
    token = get_api_read_access_token()
    if not token:
        raise TMDBAPIError(
            "TMDB API token is not configured. Please add "
            "TMDB_API_READ_ACCESS_TOKEN to your Streamlit secrets."
        )
    return {"Authorization": f"Bearer {token}", "accept": "application/json"}


def _pick_best_poster(posters: list[dict[str, Any]]) -> Optional[str]:
    """Choose the best poster from the /images endpoint response.

    Preference order:
      1. English posters (``iso_639_1 == 'en'``)
      2. Language-neutral posters (``iso_639_1 in (None, '', 'null')``)
      3. Anything else

    Within each group, rank by vote_average (then vote_count, then width).
    """
    if not posters:
        return None

    def language_rank(p: dict[str, Any]) -> int:
        lang = p.get("iso_639_1")
        if lang == "en":
            return 0
        if lang in (None, "", "null"):
            return 1
        return 2

    def quality(p: dict[str, Any]) -> tuple[float, float, int]:
        return (
            float(p.get("vote_average") or 0),
            float(p.get("vote_count") or 0),
            int(p.get("width") or 0),
        )

    best = sorted(posters, key=lambda p: (language_rank(p), tuple(-q for q in quality(p))))[0]
    return best.get("file_path")


def _fetch_poster_path_from_images(movie_id: int, headers: dict[str, str]) -> Optional[str]:
    """Fallback: GET /movie/{id}/images and pick the best poster_path."""
    data = _get_json(
        f"{TMDB_BASE_URL}/movie/{movie_id}/images",
        headers,
        params={"include_image_language": "en,null"},
    )
    if not data:
        return None
    return _pick_best_poster(data.get("posters") or [])


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_movie_details(movie_id: int) -> Optional[dict[str, Any]]:
    """Fetch movie details from TMDB using the movie ID.

    Uses ``/movie/{id}`` first; if no ``poster_path`` is returned, falls back
    to ``/movie/{id}/images`` (en + language-neutral) to find a poster.

    Only successful results are cached — exceptions propagate so a temporary
    network failure is retried on the next interaction.
    """
    headers = _auth_headers()

    try:
        movie_id_int = int(movie_id)
    except (TypeError, ValueError):
        raise TMDBAPIError(f"Invalid movie ID: {movie_id!r}") from None

    data = _get_json(f"{TMDB_BASE_URL}/movie/{movie_id_int}", headers)
    if data is None:
        return None

    poster_path = data.get("poster_path")
    if not poster_path:
        try:
            poster_path = _fetch_poster_path_from_images(movie_id_int, headers)
        except TMDBAPIError:
            poster_path = None  # fallback failed; still show metadata

    release_date = data.get("release_date") or ""
    release_year = release_date[:4] if release_date else "N/A"

    return {
        "title": data.get("title") or "Unknown title",
        "overview": data.get("overview") or "No overview available.",
        "release_year": release_year,
        "genres": [g.get("name", "") for g in data.get("genres", [])],
        "rating": data.get("vote_average", "N/A"),
        "runtime": data.get("runtime"),
        "poster_url": (f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None),
        "tmdb_url": f"https://www.themoviedb.org/movie/{movie_id_int}",
    }


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_poster_bytes(poster_url: str) -> Optional[bytes]:
    """Download poster image bytes through the Python backend.

    Using bytes (instead of a browser-side remote URL) makes the app robust
    when the TMDB image CDN/network is flaky. Only successful downloads (and
    genuine 404s) are cached; network failures raise TMDBAPIError so they are
    NOT cached and will be retried on the next interaction.
    """
    if not poster_url:
        return None

    last_error: Optional[Exception] = None
    response: Optional[requests.Response] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _SESSION.get(poster_url, timeout=REQUEST_TIMEOUT_SECONDS)
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if response is None:
        raise TMDBAPIError("Could not load the poster image. Please try again.") from None
    if response.status_code == 404:
        return None  # poster genuinely gone — safe to cache as None
    if response.status_code != 200:
        raise TMDBAPIError(f"Poster image request failed (status {response.status_code}).")

    return response.content
