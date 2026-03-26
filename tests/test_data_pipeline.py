"""
Unit tests for Vanderbilt Database Assistant
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json

from config import Config, get_config, validate_config
from utils import (
    validate_csv,
    RateLimiter,
    calculate_data_quality,
    ValidationResult,
    parse_ai_response,
    _names_match,
    build_required_database_url,
    rank_databases_from_spreadsheet,
    verify_prd_candidates,
    RESEARCH_GUIDES_FALLBACK_URL,
)


# ==================== CONFIG TESTS ====================

class TestConfig:
    """Test configuration management"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        cfg = get_config()
        assert cfg.SEARCH_TOP_K == 5
        assert cfg.LLM_MODEL == "gpt-4.1-mini"
        assert cfg.CSV_MIN_ROWS == 100
    
    def test_config_validation_passes(self):
        """Test valid configuration passes validation"""
        assert validate_config() is True
    
    def test_config_bounds(self):
        """Test configuration bounds"""
        cfg = get_config()
        assert 0 <= cfg.LLM_TEMPERATURE <= 2
        assert 100 <= cfg.LLM_MAX_TOKENS <= 4000
        assert cfg.RATE_LIMIT_PER_MINUTE >= 1


# ==================== DATA VALIDATION TESTS ====================

class TestCSVValidation:
    """Test CSV data validation"""
    
    def create_valid_df(self):
        """Create a valid test DataFrame"""
        return pd.DataFrame({
            'ID': [1, 2, 3],
            'Name': ['JSTOR', 'ProQuest', 'Database3'],
            'Description': ['Full-text journals', 'Academic database', 'Research tool'],
            'URL': ['https://jstor.org', 'https://proquest.com', 'https://db3.com'],
            'Last_Updated': ['2026-03-01', '2026-03-01', '2026-03-01'],
            'Primary_Library': ['Central', 'Business', 'Law'],
            'Alt_Names': ['Journal Storage', 'PQ', 'Database 3'],
            'Unused1': ['', '', ''],
            'Friendly_URL': ['jstor', 'proquest', 'db3'],
            'Subjects': ['Humanities', 'Business', 'Law'],
            'Unused2': ['', '', ''],
            'More_Info': ['', '', '']
        })
    
    def test_validate_valid_csv(self):
        """Test validation passes for valid CSV"""
        df = self.create_valid_df()
        result = validate_csv(df)
        assert result.is_valid is True
        assert result.record_count == 3
        assert len(result.errors) == 0
    
    def test_validate_rejects_missing_columns(self):
        """Test validation rejects CSV with missing columns"""
        df = pd.DataFrame({
            'ID': [1, 2],
            'Name': ['DB1', 'DB2']
        })
        result = validate_csv(df)
        assert result.is_valid is False
        assert any('Missing required columns' in e for e in result.errors)
    
    def test_validate_warns_few_records(self):
        """Test validation warns if record count too low"""
        df = self.create_valid_df().iloc[:2]  # Only 2 records
        result = validate_csv(df)
        assert any('below expected minimum' in w for w in result.warnings)
    
    def test_validate_calculates_metrics(self):
        """Test validation calculates data quality metrics"""
        df = self.create_valid_df()
        result = validate_csv(df)
        assert result.metrics['total_records'] == 3
        assert 'unique_subjects' in result.metrics
    
    def test_validate_empty_names_warning(self):
        """Test validation warns about empty names"""
        df = self.create_valid_df()
        df.loc[0, 'Name'] = ''
        result = validate_csv(df)
        assert any('empty Name' in w for w in result.warnings)


# ==================== RATE LIMITING TESTS ====================

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter(requests_per_minute=5)
        
        for i in range(5):
            status = limiter.is_allowed()
            assert status.is_allowed is True
            assert status.requests_in_window == i + 1
    
    def test_rate_limiter_blocks_excess(self):
        """Test rate limiter blocks excess requests"""
        limiter = RateLimiter(requests_per_minute=3)
        
        for i in range(3):
            status = limiter.is_allowed()
            assert status.is_allowed is True
        
        status = limiter.is_allowed()
        assert status.is_allowed is False
        assert status.requests_in_window == 3
    
    def test_rate_limiter_resets(self):
        """Test rate limiter resets window"""
        limiter = RateLimiter(requests_per_minute=2)
        limiter.window_start = 0  # Fake old window
        
        status = limiter.is_allowed()
        assert status.is_allowed is True


# ==================== DATA QUALITY TESTS ====================

class TestDataQuality:
    """Test data quality metrics"""
    
    def test_completeness_metric(self):
        """Test completeness metric calculation"""
        df = pd.DataFrame({
            'A': [1, 2, None],
            'B': [None, None, None],
            'C': [1, 2, 3]
        })
        
        metrics = calculate_data_quality(df)
        assert metrics['completeness']['A'] == pytest.approx(2/3, 0.01)
        assert metrics['completeness']['B'] == 0
        assert metrics['completeness']['C'] == 1.0


# ==================== AI RESPONSE PARSING TESTS ====================

class TestParseAIResponse:
    """Test parse_ai_response helper"""

    def test_parses_summary(self):
        """SUMMARY line is extracted correctly"""
        text = (
            "SUMMARY: Great databases for legal research.\n\n"
            "1. LexisNexis\nINSIGHT: Best for case law.\n"
        )
        summary, _, _ = parse_ai_response(text)
        assert summary == "Great databases for legal research."

    def test_parses_ranked_names(self):
        """Database names are extracted in ranked order"""
        text = (
            "SUMMARY: Overview.\n\n"
            "1. JSTOR\nINSIGHT: Strong humanities coverage.\n\n"
            "2. ProQuest\nINSIGHT: Broad multidisciplinary content.\n"
        )
        _, _, ranked = parse_ai_response(text)
        assert ranked == ["JSTOR", "ProQuest"]

    def test_parses_insights(self):
        """INSIGHT text is mapped to the correct database name"""
        text = (
            "SUMMARY: Overview.\n\n"
            "1. JSTOR\nINSIGHT: Strong humanities coverage.\n\n"
            "2. ProQuest\nINSIGHT: Broad multidisciplinary content.\n"
        )
        _, insights, _ = parse_ai_response(text)
        assert "JSTOR" in insights
        assert insights["JSTOR"] == "Strong humanities coverage."
        assert insights["ProQuest"] == "Broad multidisciplinary content."

    def test_handles_bold_database_names(self):
        """Bold markdown around names is stripped"""
        text = (
            "SUMMARY: Overview.\n\n"
            "1. **LexisNexis**\nINSIGHT: Primary legal resource.\n"
        )
        _, insights, ranked = parse_ai_response(text)
        assert ranked == ["LexisNexis"]
        assert "LexisNexis" in insights

    def test_handles_malformed_response_gracefully(self):
        """Malformed text returns empty structures without raising"""
        text = "Sorry, I cannot help with that request."
        summary, insights, ranked = parse_ai_response(text)
        assert summary == ""
        assert insights == {}
        assert ranked == []

    def test_handles_empty_string(self):
        """Empty string input returns empty structures"""
        summary, insights, ranked = parse_ai_response("")
        assert summary == ""
        assert insights == {}
        assert ranked == []

    def test_multiline_insight_joined(self):
        """Insight that spans multiple lines is joined into one string"""
        text = (
            "SUMMARY: Overview.\n\n"
            "1. ScienceDirect\n"
            "INSIGHT: Covers natural sciences.\n"
            "Especially strong for chemistry and biology.\n"
        )
        _, insights, _ = parse_ai_response(text)
        assert "chemistry" in insights.get("ScienceDirect", "")


class TestNamesMatch:
    """Test _names_match helper"""

    def test_exact_match(self):
        assert _names_match("JSTOR", "JSTOR") is True

    def test_case_insensitive(self):
        assert _names_match("jstor", "JSTOR") is True

    def test_substring_a_in_b(self):
        assert _names_match("JSTOR", "JSTOR: Scholarly Journals") is True

    def test_substring_b_in_a(self):
        assert _names_match("JSTOR: Scholarly Journals", "JSTOR") is True

    def test_no_match(self):
        assert _names_match("ProQuest", "LexisNexis") is False


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests"""
    
    def test_config_to_validation_flow(self):
        """Test config flows through validation"""
        cfg = get_config()
        
        df = pd.DataFrame({
            'ID': list(range(cfg.CSV_MIN_ROWS + 10)),
            'Name': [f'DB{i}' for i in range(cfg.CSV_MIN_ROWS + 10)],
            'Description': ['desc'] * (cfg.CSV_MIN_ROWS + 10),
            'URL': ['url'] * (cfg.CSV_MIN_ROWS + 10),
            'Subjects': ['subj'] * (cfg.CSV_MIN_ROWS + 10),
            'Primary_Library': ['lib'] * (cfg.CSV_MIN_ROWS + 10),
            'Alt_Names': [''] * (cfg.CSV_MIN_ROWS + 10),
            'Last_Updated': ['2026-01-01'] * (cfg.CSV_MIN_ROWS + 10),
            'Friendly_URL': [''] * (cfg.CSV_MIN_ROWS + 10),
            'Unused1': [''] * (cfg.CSV_MIN_ROWS + 10),
            'Unused2': [''] * (cfg.CSV_MIN_ROWS + 10),
            'More_Info': [''] * (cfg.CSV_MIN_ROWS + 10),
        })
        
        result = validate_csv(df)
        assert result.is_valid is True
        assert result.record_count >= cfg.CSV_MIN_ROWS


class TestPRDCompliance:
    """Test strict PRD constraints for name and URL verification."""

    def create_prd_df(self):
        return pd.DataFrame({
            'ID': [1, 2, 3, 4, 5, 6],
            'Name': ['JSTOR', 'ProQuest Central', 'PsycINFO', 'LexisNexis', 'PubMed', 'Music Index'],
            'Description': [
                'Scholarly journals in humanities and social sciences',
                'Multidisciplinary academic content and newspapers',
                'Psychology and behavioral science literature',
                'Legal research database with case law and statutes',
                'Biomedical and life sciences citations',
                'Music research, periodicals, and reference content',
            ],
            'URL': [''] * 6,
            'Last_Updated': ['2026-03-01'] * 6,
            'Primary_Library': ['Central'] * 6,
            'Alt_Names': ['', '', '', '', '', ''],
            'Unused1': ['', '', '', '', '', ''],
            'Friendly_URL': ['jstor', 'proquestcentral', '', 'lexisnexis', 'pubmed', 'musicindex'],
            'Subjects': ['Humanities', 'Multidisciplinary', 'Psychology', 'Law', 'Medicine', 'Music'],
            'Unused2': ['', '', '', '', '', ''],
            'More_Info': ['', '', '', '', '', '']
        })

    def test_build_required_database_url_uses_fallback_when_blank(self):
        assert build_required_database_url('') == RESEARCH_GUIDES_FALLBACK_URL

    def test_build_required_database_url_concatenates_exact_value(self):
        value = 'az/specialSlug-ABC_1'
        assert build_required_database_url(value) == f'https://researchguides.library.vanderbilt.edu/{value}'

    def test_verify_discards_name_not_exact_match(self):
        df = self.create_prd_df()
        candidates = [{
            'name': 'JSTOR Database',
            'description': 'x',
            'subjects': 'x',
            'friendly_url': 'jstor',
            'score': 20,
        }]
        verified = verify_prd_candidates(candidates, df)
        assert verified == []

    def test_verify_discards_friendly_url_not_same_row(self):
        df = self.create_prd_df()
        candidates = [{
            'name': 'JSTOR',
            'description': 'x',
            'subjects': 'x',
            'friendly_url': 'wrong-slug',
            'score': 20,
        }]
        verified = verify_prd_candidates(candidates, df)
        assert verified == []

    def test_rank_returns_max_five(self):
        df = self.create_prd_df()
        ranked = rank_databases_from_spreadsheet(df, 'research database academic literature law psychology medicine music humanities', top_k=10)
        assert len(ranked) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

