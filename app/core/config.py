from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_GENERATOR_MODEL: str = "qwen2.5-coder:3b"
    OLLAMA_ANALYST_MODEL: str = "gemma4:e4b"

    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore",
        case_sensitive=False
    )

settings = Settings()