"""
Utility functions for logging, validation, rate limiting, and error handling
"""

import logging
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
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
    
    # Avoid duplicate handlers on repeated imports / Streamlit reruns
    if logger.handlers:
        return logger
    
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
        metrics['unique_subjects'] = df['Subjects'].str.split(r'[,;]', regex=True).explode().str.strip().nunique()
    
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
        all_subjects = df['Subjects'].str.split(r'[,;]', regex=True).explode().str.strip().unique()
        metrics["coverage"]["unique_subjects"] = len(all_subjects)
    
    # Primary library coverage
    if 'Primary_Library' in df.columns:
        metrics["coverage"]["unique_libraries"] = df['Primary_Library'].nunique()
    
    return metrics


# ==================== SINGLETON INSTANCES ====================

_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(requests_per_minute=get_config().RATE_LIMIT_PER_MINUTE)
    return _rate_limiter


# ==================== AI RESPONSE PARSING ====================

def _names_match(name_a: str, name_b: str) -> bool:
    """Return True when two database name strings are considered equivalent.

    Performs a case-insensitive exact match as well as substring containment
    in either direction, which covers minor wording differences between the
    LLM's output and the canonical CSV name.
    """
    a = name_a.lower()
    b = name_b.lower()
    return a == b or a in b or b in a


def parse_ai_response(ai_text: str) -> Tuple[str, Dict[str, str], List[str]]:
    """Parse a structured AI response into its component parts.

    Expects the LLM to follow the format::

        SUMMARY: <brief overview>

        1. <Database Name>
        INSIGHT: <why this database is relevant>

        2. <Database Name>
        INSIGHT: <why this database is relevant>

    Returns:
        summary: The overview sentence(s).
        insights: Mapping of database name → relevance insight.
        ranked_names: Database names in the order the AI ranked them.

    Falls back gracefully when the format is not followed — callers should
    check whether the returned values are non-empty before relying on them.
    """
    summary = ""
    insights: Dict[str, str] = {}
    ranked_names: List[str] = []

    _numbered_db = re.compile(r"^\d+\.\s+\*{0,2}(.+?)\*{0,2}\s*$")

    try:
        lines = ai_text.strip().split("\n")
        current_db: Optional[str] = None
        current_insight_parts: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # SUMMARY line
            if stripped.upper().startswith("SUMMARY:"):
                summary = stripped[len("SUMMARY:"):].strip()
                current_db = None
                current_insight_parts = []

            # Numbered database line: "1. Database Name" or "1. **Database Name**"
            elif match := _numbered_db.match(stripped):
                # Save previous insight before moving to next database
                if current_db and current_insight_parts:
                    insights[current_db] = " ".join(current_insight_parts).strip()
                current_db = match.group(1).strip()
                ranked_names.append(current_db)
                current_insight_parts = []

            # INSIGHT line
            elif stripped.upper().startswith("INSIGHT:") and current_db:
                current_insight_parts = [stripped[len("INSIGHT:"):].strip()]

            # Continuation of a multi-line insight
            elif current_insight_parts:
                current_insight_parts.append(stripped)

        # Save the last insight
        if current_db and current_insight_parts:
            insights[current_db] = " ".join(current_insight_parts).strip()

    except Exception:
        pass

    return summary, insights, ranked_names


# ==================== PRD-COMPLIANT RANKING ====================

RESEARCH_GUIDES_BASE_URL = "https://researchguides.library.vanderbilt.edu/"
RESEARCH_GUIDES_FALLBACK_URL = "https://researchguides.library.vanderbilt.edu/az/databases"


def is_query_too_vague(query: str) -> bool:
    """Return True when query is too short or generic to rank reliably."""
    cleaned = re.sub(r"\s+", " ", query.strip().lower())
    if not cleaned:
        return True

    words = re.findall(r"[a-z0-9]+", cleaned)
    if len(words) < 2:
        return True

    generic_terms = {
        "help", "database", "databases", "research", "article", "articles",
        "journal", "journals", "source", "sources", "find", "need", "looking",
        "search", "topic", "paper", "papers", "info", "information",
    }
    informative_words = [w for w in words if w not in generic_terms]
    return len(informative_words) == 0


def build_required_database_url(friendly_url: str) -> str:
    """Build URL exactly from Friendly URL value or fallback when blank."""
    if friendly_url is None:
        return RESEARCH_GUIDES_FALLBACK_URL

    value = str(friendly_url)
    if value.strip() == "" or value.strip().lower() == "nan":
        return RESEARCH_GUIDES_FALLBACK_URL

    # Must concatenate exactly without normalizing casing or slug content.
    return f"{RESEARCH_GUIDES_BASE_URL}{value}"


def _tokenize_for_match(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _score_row_relevance(row: pd.Series, query_tokens: List[str]) -> int:
    """Compute a deterministic relevance score from spreadsheet metadata only."""
    name_tokens = set(_tokenize_for_match(row.get("Name", "")))
    desc_tokens = set(_tokenize_for_match(row.get("Description", "")))
    subj_tokens = set(_tokenize_for_match(row.get("Subjects", "")))
    alt_tokens = set(_tokenize_for_match(row.get("Alt_Names", "")))
    more_tokens = set(_tokenize_for_match(row.get("More_Info", "")))

    score = 0
    for token in query_tokens:
        if token in name_tokens:
            score += 6
        if token in desc_tokens:
            score += 4
        if token in subj_tokens:
            score += 5
        if token in alt_tokens:
            score += 3
        if token in more_tokens:
            score += 2
    return score


def rank_databases_from_spreadsheet(df: pd.DataFrame, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank databases by evaluating all spreadsheet rows using only row metadata."""
    query_tokens = _tokenize_for_match(query)
    if not query_tokens:
        return []

    candidates: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        score = _score_row_relevance(row, query_tokens)
        if score <= 0:
            continue

        name = str(row.get("Name", ""))
        description = str(row.get("Description", ""))
        subjects = str(row.get("Subjects", ""))
        friendly_url = str(row.get("Friendly_URL", ""))

        candidates.append({
            "name": name,
            "description": description,
            "subjects": subjects,
            "friendly_url": friendly_url,
            "score": score,
        })

    candidates.sort(key=lambda item: (-item["score"], item["name"]))
    return candidates[:max(1, min(int(top_k), 5))]


def verify_prd_candidates(candidates: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Mandatory self-check: keep only candidates that exactly match spreadsheet rows."""
    verified: List[Dict[str, Any]] = []

    for candidate in candidates:
        name = str(candidate.get("name", ""))
        if name == "":
            continue

        matched = df[df["Name"].astype(str) == name]
        if matched.empty:
            continue

        row = matched.iloc[0]
        row_friendly = str(row.get("Friendly_URL", ""))
        if row_friendly.strip().lower() == "nan":
            row_friendly = ""

        # Mandatory check: candidate friendly URL must come from same exact row.
        candidate_friendly = str(candidate.get("friendly_url", ""))
        if candidate_friendly.strip().lower() == "nan":
            candidate_friendly = ""

        if candidate_friendly != row_friendly:
            continue

        description = str(row.get("Description", ""))
        subjects = str(row.get("Subjects", ""))
        url = build_required_database_url(row_friendly)

        verified.append({
            "name": name,
            "description": description,
            "subjects": subjects,
            "friendly_url": row_friendly,
            "url": url,
            "score": int(candidate.get("score", 0)),
        })

    return verified


def build_query_matched_explanation(query: str, description: str, subjects: str) -> str:
    """Generate concise explanation grounded in Column C and row metadata."""
    desc = str(description).strip()
    desc_compact = re.sub(r"\s+", " ", desc)
    if len(desc_compact) > 220:
        desc_compact = desc_compact[:217].rstrip() + "..."

    subjects_compact = re.sub(r"\s+", " ", str(subjects).strip())
    if subjects_compact and subjects_compact.lower() != "nan":
        return f"Column C notes: {desc_compact} This aligns with your query on '{query}' through subject coverage in {subjects_compact}."
    return f"Column C notes: {desc_compact} This aligns with your query on '{query}'."
