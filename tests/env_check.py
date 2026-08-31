import os 
from dotenv import load_dotenv


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_DIR)
KEY = os.getenv("GEMINI_API_KEY")

print(KEY)