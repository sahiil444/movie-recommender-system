import sys
import ast
import pandas as pd

from src.movie_recommender_system.logger import logging
from src.movie_recommender_system.exception import CustomException

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1]

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("N"):
        return wordnet.NOUN

    elif tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN

def lemmatize_text(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))

    filtered_word = []

    for word in words:
        if word not in stop_words:
            filtered_word.append(word)

    lemmatized_words = []

    for word in filtered_word:
        pos = get_wordnet_pos(word)
        lemma = lemmatizer.lemmatize(word, pos)
        lemmatized_words.append(lemma)

    return " ".join(lemmatized_words)


def preprocess_data(merged_df):
    try:
        logging.info("Starting data preprocessing")

        required_columns = [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew"
        ]

        merged_df = merged_df[required_columns]

        merged_df = merged_df.drop_duplicates()
        merged_df = merged_df.dropna()

        def convert1(text):
            list_genres = []
            text = ast.literal_eval(text)

            for item in text:
                list_genres.append(item["name"])

            return list_genres

        def convert2(text):
            list_keywords = []
            text = ast.literal_eval(text)

            for item in text:
                list_keywords.append(item["name"])

            return list_keywords

        def convert3(text):
            list_cast = []
            text = ast.literal_eval(text)
            count = 0

            for item in text:
                if count > 7:
                    break
                else:
                    list_cast.append(item["name"])
                    count += 1

            return list_cast

        def convert4(text):
            list_director = []
            text = ast.literal_eval(text)

            for item in text:
                if item["job"] == "Director":
                    list_director.append(item["name"])

            return list_director

        merged_df["genres"] = merged_df["genres"].apply(convert1)
        merged_df["keywords"] = merged_df["keywords"].apply(convert2)
        merged_df["cast"] = merged_df["cast"].apply(convert3)
        merged_df["crew"] = merged_df["crew"].apply(convert4)

        merged_df["cast"] = merged_df["cast"].apply(lambda x: [item.strip().replace(" ", "") for item in x])
        merged_df["crew"] = merged_df["crew"].apply(lambda x: [item.strip().replace(" ", "") for item in x])
        merged_df["keywords"] = merged_df["keywords"].apply(lambda x: [item.strip().replace(" ", "") for item in x])
        merged_df["genres"] = merged_df["genres"].apply(lambda x: [item.strip().replace(" ", "") for item in x])

        merged_df["genres"] = merged_df["genres"].apply(lambda x: " ".join(x))
        merged_df["keywords"] = merged_df["keywords"].apply(lambda x: " ".join(x))
        merged_df["cast"] = merged_df["cast"].apply(lambda x: " ".join(x))
        merged_df["crew"] = merged_df["crew"].apply(lambda x: " ".join(x))

        merged_df["tags"] = (
            merged_df["overview"] + " " + 
            merged_df["genres"] + " " + 
            merged_df["keywords"] + " " + 
            merged_df["cast"] + " " + 
            merged_df["crew"] + " " +
            merged_df["title"]
        )

        merged_df["tags"] = merged_df["tags"].apply(lemmatize_text)

        new_movies2 = merged_df[["movie_id", "title", "tags"]].copy()
        new_movies2["tags"] = new_movies2["tags"].str.lower()
        

        logging.info("Data preprocessing completed successfully")

        return new_movies2

    except Exception as e:
        logging.info("Error occurred during data preprocessing")
        raise CustomException(e, sys)