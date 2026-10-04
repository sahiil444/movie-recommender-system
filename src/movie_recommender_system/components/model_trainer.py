import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.movie_recommender_system.logger import logging
from src.movie_recommender_system.exception import CustomException


def train_model(new_movies2):
    try:
        logging.info("Starting model training")

        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )

        tfidf_vectors = tfidf.fit_transform(new_movies2["tags"])

        similarity_matrix = cosine_similarity(tfidf_vectors)

        logging.info("TF-IDF and cosine similarity completed successfully")

        return tfidf, similarity_matrix

    except Exception as e:
        logging.info("Error occurred during model training")
        raise CustomException(e, sys)


def recommend(movie_name, new_movies2, similarity_matrix):
    try:
        logging.info(f"Generating recommendations for movie: {movie_name}")

        movie_index = new_movies2[new_movies2["title"] == movie_name].index[0]
        movies_distances = similarity_matrix[movie_index]
        movies_distances = list(enumerate(movies_distances))

        top_movies_distances = sorted(
            movies_distances,
            reverse=True,
            key=lambda x: x[1]
        )

        recommendations = []

        for item in top_movies_distances[1:7]:
            index = item[0]
            recommendations.append(new_movies2.iloc[index]["title"])

        logging.info(f"Recommendations generated successfully for: {movie_name}")

        return recommendations

    except Exception as e:
        logging.info("Error occurred while generating recommendations")
        raise CustomException(e, sys)