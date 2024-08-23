from dotenv import dotenv_values
from pathlib import Path


BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'data'
MODEL_PATH = BASE_PATH / 'models'
TASK_PATH = BASE_PATH / 'tasks'

