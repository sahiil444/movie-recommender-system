import os
import sys
import pandas as pd
from src.movie_recommender_system.logger import logging
from src.movie_recommender_system.exception import CustomException



def load_data():
    try:
        logging.info("Starting data ingestion")

        movies_path = os.path.join(
            "..","data","raw","tmdb_5000_movies.csv"
        )

        credits_path = os.path.join(
            "..","data","raw","tmdb_5000_credits.csv"
        )

        movies_df = pd.read_csv(movies_path)
        credits_df = pd.read_csv(credits_path)

        logging.info("Movies and credits datasets loaded successfully")

        return movies_df,credits_df
    
    except Exception as e:
        logging.info("Error occurred while loading datasets")
        raise CustomException(e,sys)


def validate_data(movie_df,credits_df):
    try:
        logging.info("Starting data validation")

        if movie_df.empty:
            raise ValueError("Movies dataset is empty")
        if credits_df.empty:
            raise ValueError("Movies dataset is empty")

        required_movies_columns = ["id", "title"]
        required_credits_columns = ["movie_id", "title"]

        for column in required_movies_columns:
            if column not in movie_df.columns:
                raise ValueError(f"Missing column in movies dataset: {column}")

        for column in required_credits_columns:
            if column not in credits_df.columns:
                raise ValueError(f"Missing column in movies dataset: {column}")

        logging.info("Data validation completed successfully")

    except Exception as e:
        logging.info("Error occurred during data validation")
        raise CustomException(e,sys)


def merge_data(movies_df,credits_df):
    try:

        logging.info("Starting data merging")
        credits_df = credits_df.drop(columns=["title"])
        movies_df = movies_df.merge(
            credits_df,
            left_on = "id",
            right_on = "movie_id"
        )


        logging.info("Data merging completed successfully")
        return movies_df
    

    except Exception as e:
        logging.info("Error occurred during data merging")
        raise CustomException(e,sys)
