import os
from src.movie_recommender_system.utils import load_object


class PredictionPipeline:

    def __init__(self):
        self.processed_df_path = "artifacts/processed_df.pkl"
        self.tfidf_path = "artifacts/tfidf.pkl"
        self.similarity_matrix_path = "artifacts/similarity_matrix.pkl"

        self.processed_df = load_object(self.processed_df_path)
        self.tfidf = load_object(self.tfidf_path)
        self.similarity_matrix = load_object(self.similarity_matrix_path)


    def predict(self, movie_name):
        movie_index = self.processed_df[self.processed_df["title"] == movie_name].index[0]
        movies_distances = self.similarity_matrix[movie_index]

        movies_distances = list(enumerate(movies_distances))

        top_movies_distances = sorted(
            movies_distances,
            reverse=True,
            key=lambda x: x[1]
        )

        recommendations = []

        for item in top_movies_distances[1:7]:
            index = item[0]
            recommendations.append(
            self.processed_df.iloc[index]["title"]
        )

        return recommendations