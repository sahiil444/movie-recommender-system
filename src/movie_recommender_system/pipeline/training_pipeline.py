import sys
from src.movie_recommender_system.logger import logging
from src.movie_recommender_system.exception import CustomException

from src.movie_recommender_system.components.data_ingestion import(
    load_data,validate_data,merge_data
)
from src.movie_recommender_system.components.data_preprocessing import preprocess_data
from src.movie_recommender_system.components.model_trainer import train_model
from src.movie_recommender_system.utils import save_object



def run_training_pipeline():
    try:
        logging.info("Starting training pipeline")

        # Data ingestion
        movies_df, credits_df = load_data()
        validate_data(movies_df, credits_df)
        merged_df = merge_data(movies_df, credits_df)

        # Data preprocessing
        processed_df = preprocess_data(merged_df)

        # Model training
        tfidf, similarity_matrix = train_model(processed_df)
        save_object("artifacts/processed_df.pkl", processed_df)
        save_object("artifacts/tfidf.pkl", tfidf)
        save_object("artifacts/similarity_matrix.pkl", similarity_matrix)

        logging.info("Training pipeline completed successfully & Model artifacts saved successfully")

        return processed_df, tfidf, similarity_matrix

    except Exception as e:
        logging.info("Error occurred in training pipeline")
        raise CustomException(e, sys)