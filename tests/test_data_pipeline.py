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
    CostTracker, 
    calculate_data_quality,
    ValidationResult
)


# ==================== CONFIG TESTS ====================

class TestConfig:
    """Test configuration management"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        cfg = get_config()
        assert cfg.SEARCH_TOP_K == 5
        assert cfg.LLM_MODEL == "gpt-4o-mini"
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


# ==================== COST TRACKING TESTS ====================

class TestCostTracker:
    """Test API cost tracking"""
    
    def test_cost_calculation(self):
        """Test cost calculation for API call"""
        tracker = CostTracker()
        cost = tracker.calculate_cost(
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert cost.model == "gpt-4o-mini"
        assert cost.total_tokens == 150
        assert cost.cost_usd > 0
    
    def test_cost_accumulation(self):
        """Test cost tracking accumulates correctly"""
        tracker = CostTracker()
        
        tracker.calculate_cost("gpt-4o-mini", 100, 50)
        tracker.calculate_cost("gpt-4o-mini", 100, 50)
        
        stats = tracker.get_stats()
        assert stats['call_count'] == 2
        assert "$" in stats['total_cost_usd']
    
    def test_cost_logging_format(self):
        """Test cost logs are valid JSON"""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            temp_file = f.name
        
        try:
            tracker = CostTracker()
            tracker.cost_log = Path(temp_file)
            
            cost = tracker.calculate_cost("gpt-4o-mini", 100, 50)
            tracker.log_cost(cost)
            
            with open(temp_file, 'r') as f:
                line = f.readline()
                data = json.loads(line)
                assert 'model' in data
                assert 'cost_usd' in data
        finally:
            Path(temp_file).unlink()


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
        })
        
        result = validate_csv(df)
        assert result.is_valid is True
        assert result.record_count >= cfg.CSV_MIN_ROWS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
