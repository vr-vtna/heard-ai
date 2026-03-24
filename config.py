"""
Configuration management for Vanderbilt Database Assistant
Centralized settings for all components
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Config:
    """Application configuration with environment variable overrides"""

    # ==================== CACHE ====================
    CSV_CACHE_TTL: int = int(os.getenv("CSV_CACHE_TTL", "3600"))  # 1 hour
    CHROMADB_CACHE_TTL: int = int(os.getenv("CHROMADB_CACHE_TTL", "3600"))

    # ==================== SEARCH ====================
    SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K", "5"))
    SEARCH_BATCH_SIZE: int = int(os.getenv("SEARCH_BATCH_SIZE", "100"))

    # ==================== AI/LLM ====================
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))

    # ==================== RATE LIMITING ====================
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

    # ==================== DATA ====================
    CSV_GLOB_PATTERN: str = os.getenv("CSV_GLOB_PATTERN", "data/az_database_list_*.csv")
    CSV_MIN_ROWS: int = int(os.getenv("CSV_MIN_ROWS", "100"))
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", "chroma_data")

    # ==================== LOGGING ====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")

    # ==================== SESSION ====================
    MAX_SESSION_HISTORY: int = int(os.getenv("MAX_SESSION_HISTORY", "50"))
    MAX_FEEDBACK_ITEMS: int = int(os.getenv("MAX_FEEDBACK_ITEMS", "1000"))

    # ==================== CSV VALIDATION ====================
    REQUIRED_CSV_COLUMNS: tuple = (
        "ID",
        "Name",
        "Description",
        "URL",
        "Subjects"
    )
    EXPECTED_CSV_COLUMN_COUNT: int = 12

    # ==================== COST TRACKING ====================
    TRACK_API_COSTS: bool = os.getenv("TRACK_API_COSTS", "true").lower() == "true"
    COST_LOG_FILE: str = os.getenv("COST_LOG_FILE", "logs/api_costs.log")

    # ==================== VANDERBILT BRANDING ====================
    PRIMARY_COLOR: str = "#866D4B"  # Vanderbilt gold
    SECONDARY_COLOR: str = "#6F5A3E"  # Darker gold
    TEXT_COLOR: str = "#000000"
    BG_COLOR: str = "#FFFFFF"
    SECONDARY_BG_COLOR: str = "#F8F9FA"


# Create singleton instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config


def validate_config() -> bool:
    """Validate configuration consistency"""
    cfg = get_config()
    
    errors = []
    
    if cfg.SEARCH_TOP_K < 1 or cfg.SEARCH_TOP_K > 20:
        errors.append("SEARCH_TOP_K must be between 1 and 20")
    
    if cfg.LLM_TEMPERATURE < 0 or cfg.LLM_TEMPERATURE > 2:
        errors.append("LLM_TEMPERATURE must be between 0 and 2")
    
    if cfg.LLM_MAX_TOKENS < 100 or cfg.LLM_MAX_TOKENS > 4000:
        errors.append("LLM_MAX_TOKENS must be between 100 and 4000")
    
    if cfg.RATE_LIMIT_PER_MINUTE < 1:
        errors.append("RATE_LIMIT_PER_MINUTE must be at least 1")
    
    if errors:
        for error in errors:
            print(f"⚠️  Config error: {error}")
        return False
    
    return True
