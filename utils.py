"""
Utility functions for logging, validation, rate limiting, and error handling
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque
import time
from dataclasses import dataclass
import pandas as pd

from config import get_config


# ==================== LOGGING ====================

def setup_logging() -> logging.Logger:
    """Initialize structured logging"""
    cfg = get_config()
    
    # Create logs directory
    log_dir = Path(cfg.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger("heard-ai")
    logger.setLevel(cfg.LOG_LEVEL)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / cfg.LOG_FILE)
    file_handler.setLevel(cfg.LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(cfg.LOG_LEVEL)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


def _json_safe_value(value: Any) -> Any:
    """Convert common non-JSON-native values (numpy/pandas) to plain Python types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {k: _json_safe_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(v) for v in value]

    return value


def log_event(event_name: str, **kwargs) -> None:
    """Log structured events as JSON"""
    event = {
        "event": event_name,
        "timestamp": datetime.now().isoformat(),
        **{k: _json_safe_value(v) for k, v in kwargs.items()}
    }
    logger.info(json.dumps(event, default=str))


# ==================== DATA VALIDATION ====================

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    record_count: int
    errors: list
    warnings: list
    metrics: Dict[str, Any]


def validate_csv(df: pd.DataFrame) -> ValidationResult:
    """Validate CSV structure and content"""
    cfg = get_config()
    errors = []
    warnings = []
    metrics = {}
    
    # Check column count (warn only; required columns check determines validity)
    if len(df.columns) < cfg.EXPECTED_CSV_COLUMN_COUNT:
        warnings.append(
            f"CSV has {len(df.columns)} columns, "
            f"expected around {cfg.EXPECTED_CSV_COLUMN_COUNT}"
        )
    
    # Check required columns exist (after renaming)
    missing_cols = [col for col in cfg.REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for empty critical fields
    if 'Name' in df.columns:
        empty_names = (df['Name'].isna().sum() + 
                      (df['Name'].str.strip() == "").sum())
        if empty_names > 0:
            warnings.append(f"{empty_names} records have empty Name field")
    
    if 'Description' in df.columns:
        empty_desc = (df['Description'].isna().sum() + 
                     (df['Description'].str.strip() == "").sum())
        if empty_desc > 0:
            warnings.append(f"{empty_desc} records have empty Description field")
    
    # Calculate metrics
    metrics['total_records'] = len(df)
    metrics['missing_names'] = df['Name'].isna().sum() if 'Name' in df.columns else 0
    metrics['missing_descriptions'] = df['Description'].isna().sum() if 'Description' in df.columns else 0
    
    if 'Subjects' in df.columns:
        metrics['unique_subjects'] = df['Subjects'].str.split(',').explode().nunique()
    
    if 'Last_Updated' in df.columns:
        metrics['oldest_update'] = str(df['Last_Updated'].min())
        metrics['newest_update'] = str(df['Last_Updated'].max())
    
    # Check minimum row count
    if len(df) < cfg.CSV_MIN_ROWS:
        warnings.append(
            f"Database count ({len(df)}) is below expected "
            f"minimum ({cfg.CSV_MIN_ROWS})"
        )
    
    is_valid = len(errors) == 0
    
    result = ValidationResult(
        is_valid=is_valid,
        record_count=len(df),
        errors=errors,
        warnings=warnings,
        metrics=metrics
    )
    
    # Log validation result
    log_event(
        "csv_validated",
        is_valid=is_valid,
        record_count=len(df),
        errors_count=len(errors),
        warnings_count=len(warnings),
        metrics=metrics
    )
    
    return result


# ==================== RATE LIMITING ====================

@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    requests_in_window: int
    max_requests: int
    is_allowed: bool
    reset_in_seconds: float


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int = 10):
        self.max_requests = requests_per_minute
        self.window_start = time.time()
        self.requests = deque()
    
    def is_allowed(self) -> RateLimitStatus:
        """Check if request is allowed"""
        now = time.time()
        window_end = self.window_start + 60
        
        # Reset window if needed
        if now >= window_end:
            self.window_start = now
            self.requests.clear()
        
        # Remove old requests outside current window
        while self.requests and self.requests[0] < self.window_start:
            self.requests.popleft()
        
        allowed = len(self.requests) < self.max_requests
        
        if allowed:
            self.requests.append(now)
        
        reset_seconds = max(0, window_end - now)
        
        return RateLimitStatus(
            requests_in_window=len(self.requests),
            max_requests=self.max_requests,
            is_allowed=allowed,
            reset_in_seconds=reset_seconds
        )


# ==================== COST TRACKING ====================

@dataclass
class APICallCost:
    """Cost of an API call"""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str


class CostTracker:
    """Track API usage costs"""
    
    # Pricing as of March 2026 (gpt-4o-mini)
    PRICING = {
        "gpt-4o-mini": {
            "input": 0.00015 / 1000,      # $0.15 per 1M input tokens
            "output": 0.0006 / 1000,     # $0.60 per 1M output tokens
        }
    }
    
    def __init__(self):
        cfg = get_config()
        self.cost_log = Path(cfg.COST_LOG_FILE)
        self.cost_log.parent.mkdir(exist_ok=True)
        self.total_cost = 0.0
        self.call_count = 0
    
    def calculate_cost(
        self, 
        model: str, 
        prompt_tokens: int, 
        completion_tokens: int
    ) -> APICallCost:
        """Calculate cost of API call"""
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])
        
        input_cost = prompt_tokens * pricing["input"]
        output_cost = completion_tokens * pricing["output"]
        total_cost = input_cost + output_cost
        
        self.total_cost += total_cost
        self.call_count += 1
        
        return APICallCost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=total_cost,
            timestamp=datetime.now().isoformat()
        )
    
    def log_cost(self, cost: APICallCost) -> None:
        """Log API cost to file"""
        if not get_config().TRACK_API_COSTS:
            return
        
        with open(self.cost_log, 'a') as f:
            f.write(json.dumps({
                "model": cost.model,
                "tokens": cost.total_tokens,
                "cost_usd": cost.cost_usd,
                "timestamp": cost.timestamp
            }) + "\n")
        
        log_event(
            "api_call",
            model=cost.model,
            prompt_tokens=cost.prompt_tokens,
            completion_tokens=cost.completion_tokens,
            cost_usd=f"${cost.cost_usd:.4f}",
            total_cost_usd=f"${self.total_cost:.2f}",
            call_count=self.call_count
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost statistics"""
        return {
            "total_cost_usd": f"${self.total_cost:.2f}",
            "call_count": self.call_count,
            "avg_cost_per_call": f"${self.total_cost / max(1, self.call_count):.4f}"
        }


# ==================== DATA QUALITY METRICS ====================

def calculate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate data quality metrics"""
    metrics = {
        "total_records": len(df),
        "completeness": {},
        "coverage": {}
    }
    
    # Completeness metrics
    for col in df.columns:
        metrics["completeness"][col] = 1 - (df[col].isna().sum() / len(df))
    
    # Subject coverage
    if 'Subjects' in df.columns:
        all_subjects = df['Subjects'].str.split(',', expand=True).stack().str.strip().unique()
        metrics["coverage"]["unique_subjects"] = len(all_subjects)
    
    # Primary library coverage
    if 'Primary_Library' in df.columns:
        metrics["coverage"]["unique_libraries"] = df['Primary_Library'].nunique()
    
    return metrics


# ==================== SINGLETON INSTANCES ====================

_rate_limiter: Optional[RateLimiter] = None
_cost_tracker: Optional[CostTracker] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(requests_per_minute=get_config().RATE_LIMIT_PER_MINUTE)
    return _rate_limiter


def get_cost_tracker() -> CostTracker:
    """Get or create cost tracker instance"""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
