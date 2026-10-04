import os
import pickle


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def save_object(file_path, obj):
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)



def load_object(file_path):
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)

    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)