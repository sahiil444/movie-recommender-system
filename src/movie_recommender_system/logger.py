import logging
import os
from datetime import datetime

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FOLDER = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(LOG_FOLDER, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_FOLDER, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


