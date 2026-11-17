from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    index_path: Path = Path("index.json")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()