# 📚 Vanderbilt Database Assistant

AI-powered chatbot to help users discover relevant databases from Vanderbilt University Library's collection.

## Features

- 🔍 **Semantic Search**: Intelligent database discovery across 500+ resources using ChromaDB
- 🤖 **AI-Powered Recommendations**: Personalized database suggestions via Vanderbilt Amplify API
- 📊 **Subject Filtering**: Browse and filter databases by academic discipline
- 🔄 **Weekly Automatic Data Refresh**: GitHub Actions-scheduled database updates
- 📱 **Mobile-Responsive Design**: Works seamlessly on desktop and mobile devices
- 🎨 **Vanderbilt Branding**: Custom UI with institutional colors and typography
- 💬 **User Feedback**: Built-in feedback mechanism to improve recommendations

## Quick Start

### Prerequisites

- Python 3.8+
- Vanderbilt Amplify API token
- Git

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Secrets

Create `.streamlit/secrets.toml`:

```toml
AMPLIFY_TOKEN = "your-amplify-token-here"
```

Alternatively, set as environment variable:
```bash
export AMPLIFY_TOKEN="your-amplify-token"
```

### 3. Run the App

```bash
streamlit run streamlit_app.py
```

The app will open automatically at `http://localhost:8501`

## Project Structure

```
heard-ai/
├── streamlit_app.py              # Main application
├── requirements.txt              # Python dependencies (pinned versions)
├── README.md                     # This file
├── LICENSE                       # Repository license
├── .streamlit/
│   ├── config.toml              # Streamlit theme & server config
│   └── secrets.toml             # API keys (git-ignored)
├── .github/
│   └── workflows/
│       └── refresh_database.yml # Weekly data refresh automation
├── scripts/
│   └── refresh_data.py          # Manual data refresh utility
└── data/
    └── az_database_list_*.csv   # Vanderbilt database exports
```

## Data Setup

Place Vanderbilt database CSV files in the `data/` directory:

```bash
data/az_database_list_20260324.csv
```

**Expected CSV Columns:**
1. ID
2. Name
3. Description
4. URL
5. Last_Updated
6. Primary_Library
7. Alt_Names
8. (unused)
9. Friendly_URL
10. Subjects
11. (unused)
12. More_Info

The app automatically discovers the most recent timestamped CSV file.

## Manual Data Refresh

```bash
# Using local copy
python scripts/refresh_data.py

# With remote URL
CSV_SOURCE_URL="https://your-url.com/databases.csv" python scripts/refresh_data.py
```

Older CSV files (> 4 weeks) are automatically cleaned up.

## Architecture

### Data Pipeline
- **Loading**: Cached CSV parsing with intelligent glob-based file discovery
- **Indexing**: ChromaDB vector database with semantic embeddings
- **Search**: Semantic similarity queries across rich metadata

### AI Component
- **Model**: GPT-4o-mini via Vanderbilt Amplify API
- **Prompting**: Strict guidelines to prevent hallucination
- **Context**: Top 5 semantic match results with full metadata

### Frontend
- **Framework**: Streamlit
- **Styling**: Custom CSS with Vanderbilt branding (#866D4B gold, #000000 black)
- **Session State**: Persistent conversation history and search state

## Key Functions

| Function | Purpose |
|----------|---------|
| `load_databases()` | Load and standardize CSV data with caching |
| `init_vector_search()` | Initialize ChromaDB collection with database metadata |
| `search_databases()` | Execute semantic search queries |
| `generate_ai_response()` | Create personalized recommendations via Amplify API |

## Configuration

### Streamlit Config (`.streamlit/config.toml`)
- Theme colors and fonts
- Server port (8501)
- Security settings (XSRF protection, CORS disabled)
- Headless mode for production

### Environment Variables
- `AMPLIFY_TOKEN` — **Required** for Vanderbilt Amplify API access
- `CSV_SOURCE_URL` — Optional URL for remote database exports

## Deployment

### Streamlit Cloud
1. Push repo to GitHub
2. Create new app at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Configure repository and branch
4. Add `AMPLIFY_TOKEN` in Secrets

### Docker
```bash
docker build -t heard-ai .
docker run -p 8501:8501 -e AMPLIFY_TOKEN="your-token" heard-ai
```

### GitHub Pages / Custom Server
See `.github/workflows/refresh_database.yml` for CI/CD automation.

## Limitations & Considerations

- **Embedding Model**: ChromaDB uses built-in Onnx embeddings (no external API key needed)
- **Rate Limiting**: Amplify API rate limits may apply during high traffic
- **CSV Format**: Strict column ordering required; run validation before deployment
- **Cache TTL**: Database CSV cached for 1 hour; modify `@st.cache_data(ttl=3600)` as needed

## Support & Maintenance

**Issues or questions?** Contact: [library@vanderbilt.edu](mailto:library@vanderbilt.edu)

**Automated Updates**: Database refreshes every Sunday at 2 AM UTC via GitHub Actions

**Manual Trigger**: Run GitHub Actions workflow manually from the repository's Actions tab

## Advanced Usage

### Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html

# Specific test class
pytest tests/test_data_pipeline.py::TestCSVValidation -v
```

### Configuration Overrides

Key environment variables (all optional):

```bash
# Search settings
export SEARCH_TOP_K=10                    # Results per search (default: 5)

# AI model behavior
export LLM_TEMPERATURE=0.5                # Creativity 0-2 (default: 0.7)
export LLM_MAX_TOKENS=500                 # Max response length (default: 800)

# Service limits
export RATE_LIMIT_PER_MINUTE=20           # API rate limit (default: 10/min)
export CSV_CACHE_TTL=7200                 # Cache timeout in seconds (default: 3600)

# Observability
export LOG_LEVEL=DEBUG                    # Logging verbosity (default: INFO)
export TRACK_API_COSTS=true               # Track API spending (default: false)
export RATE_LIMIT_ENABLED=true            # Enable rate limiting (default: true)

# Example: Development mode
LOG_LEVEL=DEBUG RATE_LIMIT_PER_MINUTE=100 streamlit run streamlit_app.py
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
streamlit run streamlit_app.py

# Monitor logs in real-time
tail -f logs/app.log

# Check API cost logs
tail -f logs/api_costs.log

# Test CSV validation
python -c "
import pandas as pd
from utils import validate_csv
df = pd.read_csv('data/az_database_list_*.csv')
print(validate_csv(df))
"

# Check rate limiter status
python -c "
from utils import get_rate_limiter
limiter = get_rate_limiter()
for i in range(15):
    status = limiter.is_allowed()
    print(f'Request {i+1}: {status.is_allowed}')
"
```

### Production Checklist

- [ ] Run `pytest tests/ -v` (all pass)
- [ ] Set `AMPLIFY_TOKEN` environment variable
- [ ] Verify CSV in `data/` directory
- [ ] Set `LOG_LEVEL=INFO` (production)
- [ ] Set `RATE_LIMIT_PER_MINUTE=5` (production)
- [ ] Set `TRACK_API_COSTS=true`
- [ ] Monitor `logs/app.log` and `logs/api_costs.log`

## License

See [LICENSE](LICENSE) for details.
