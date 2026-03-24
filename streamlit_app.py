import streamlit as st
import pandas as pd
import sqlite3
import sys

# ChromaDB requires sqlite >= 3.35; swap in pysqlite3 when available.
if sqlite3.sqlite_version_info < (3, 35, 0):
    try:
        import pysqlite3  # type: ignore
        sys.modules["sqlite3"] = pysqlite3
    except Exception:
        pass

import chromadb
from openai import OpenAI, RateLimitError, APIConnectionError
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import os
import glob
import logging

from config import get_config, validate_config
from utils import (
    logger,
    log_event,
    validate_csv,
    get_rate_limiter,
    get_cost_tracker,
    calculate_data_quality
)

# Validate config on startup
if not validate_config():
    st.error("❌ Configuration validation failed. Please check settings.")
    st.stop()

cfg = get_config()

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Vanderbilt Database Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Vanderbilt branding
st.markdown(f"""
<style>
    /* Vanderbilt colors: Gold {cfg.PRIMARY_COLOR}, Black {cfg.TEXT_COLOR} */
    .stApp {{
        background-color: {cfg.BG_COLOR};
    }}
    
    .stButton>button {{
        background-color: {cfg.PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: {cfg.SECONDARY_COLOR};
        box-shadow: 0 4px 12px rgba(134, 109, 75, 0.3);
    }}
    
    h1 {{
        color: {cfg.TEXT_COLOR};
        font-family: 'Georgia', serif;
        font-weight: 700;
    }}
    
    h2, h3 {{
        color: {cfg.PRIMARY_COLOR};
        font-family: 'Georgia', serif;
    }}
    
    .database-card {{
        border: 2px solid {cfg.PRIMARY_COLOR};
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        background-color: {cfg.SECONDARY_BG_COLOR};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    .stExpander {{
        border: 1px solid {cfg.PRIMARY_COLOR};
        border-radius: 8px;
    }}
    
    .update-badge {{
        background-color: {cfg.PRIMARY_COLOR};
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, {cfg.PRIMARY_COLOR} 0%, {cfg.SECONDARY_COLOR} 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'positive_feedback' not in st.session_state:
    st.session_state.positive_feedback = []
if 'negative_feedback' not in st.session_state:
    st.session_state.negative_feedback = []

# ============================================
# DATA LOADING & VALIDATION
# ============================================

@st.cache_data(ttl=cfg.CSV_CACHE_TTL)
def load_databases() -> Tuple[pd.DataFrame, Optional[str]]:
    """Load and validate the most recent database CSV file"""
    try:
        # Find timestamped CSV files
        csv_files = glob.glob(cfg.CSV_GLOB_PATTERN)
        
        if not csv_files:
            error_msg = f"No database CSV found in {cfg.CSV_GLOB_PATTERN}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        latest_csv = max(csv_files, key=os.path.getctime)
        logger.info(f"Loading CSV: {latest_csv}")
        
        # Load and clean data
        df = pd.read_csv(latest_csv)
        df = df.fillna("")
        
        # Rename columns mapping (positional)
        column_mapping = {
            df.columns[0]: 'ID',
            df.columns[1]: 'Name',
            df.columns[2]: 'Description',
            df.columns[3]: 'URL',
            df.columns[4]: 'Last_Updated',
            df.columns[5]: 'Primary_Library',
            df.columns[6]: 'Alt_Names',
            df.columns[8]: 'Friendly_URL',
            df.columns[9]: 'Subjects',
            df.columns[11]: 'More_Info'
        }
        
        existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mappings)
        
        # Validate
        validation_result = validate_csv(df)
        
        if not validation_result.is_valid:
            errors_msg = "; ".join(validation_result.errors)
            logger.error(f"CSV validation failed: {errors_msg}")
            raise ValueError(f"CSV validation failed: {errors_msg}")
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"CSV warning: {warning}")
        
        log_event("csv_loaded", file=latest_csv, records=len(df))
        return df, latest_csv
    
    except FileNotFoundError as e:
        logger.error(f"CSV not found: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error loading database: {e}")
        raise


def get_last_update_date() -> str:
    """Get timestamp of last database update"""
    try:
        csv_files = glob.glob(cfg.CSV_GLOB_PATTERN)
        if csv_files:
            latest = max(csv_files, key=os.path.getctime)
            timestamp = os.path.getctime(latest)
            return datetime.fromtimestamp(timestamp).strftime('%B %d, %Y')
        return "Unknown"
    except Exception as e:
        logger.error(f"Error getting last update date: {e}")
        return "Unknown"

# ============================================
# VECTOR SEARCH
# ============================================

@st.cache_resource
def init_vector_search(_df: pd.DataFrame) -> Optional[Any]:
    """Initialize ChromaDB vector search with database content"""
    try:
        # Use persistent storage
        client = chromadb.PersistentClient(path=cfg.PERSIST_DIR)
        
        # Delete old collection if exists
        try:
            client.delete_collection("vanderbilt_databases")
        except Exception:
            pass
        
        collection = client.create_collection(
            name="vanderbilt_databases",
            metadata={"description": "Vanderbilt Database A-Z List"}
        )
        
        # Index all databases
        for idx, row in _df.iterrows():
            # Create rich search text with weighted fields
            search_text = f"""
NAME: {row.get('Name', '')}
NAME: {row.get('Name', '')}
DESCRIPTION: {row.get('Description', '')}
SUBJECTS: {row.get('Subjects', '')}
KEYWORDS: {row.get('Alt_Names', '')}
LIBRARY: {row.get('Primary_Library', '')}
INFO: {row.get('More_Info', '')}
"""
            
            # Prepare metadata
            metadata = {
                'id': str(row.get('ID', idx)),
                'name': str(row.get('Name', 'Unknown'))[:500],
                'url': str(row.get('URL', '')),
                'description': str(row.get('Description', 'No description'))[:1000],
                'subjects': str(row.get('Subjects', ''))[:500],
                'primary_library': str(row.get('Primary_Library', ''))[:200],
                'alt_names': str(row.get('Alt_Names', ''))[:300],
                'more_info': str(row.get('More_Info', ''))[:500],
                'friendly_url': str(row.get('Friendly_URL', ''))
            }
            
            collection.add(
                documents=[search_text],
                metadatas=[metadata],
                ids=[str(row.get('ID', idx))]
            )
        
        logger.info(f"ChromaDB indexed {len(_df)} databases")
        log_event("vector_search_initialized", database_count=len(_df))
        return collection
    
    except Exception as e:
        logger.exception(f"Error initializing vector search: {e}")
        raise

# ============================================
# SEARCH FUNCTIONS
# ============================================

def search_databases(
    query: str, 
    collection: Optional[Any],
    num_results: int = None
) -> Optional[Dict]:
    """Perform semantic search on database collection"""
    if num_results is None:
        num_results = cfg.SEARCH_TOP_K
    
    try:
        log_event("search_started", query=query, num_results=num_results)
        results = collection.query(
            query_texts=[query],
            n_results=num_results
        )
        
        log_event(
            "search_completed",
            query=query,
            results_count=len(results['metadatas'][0]) if results else 0
        )
        
        return results
    
    except Exception as e:
        logger.exception(f"Search error: {e}")
        log_event("search_error", query=query, error=str(e))
        return None

# ============================================
# AI RESPONSE GENERATION
# ============================================

def generate_ai_response(
    query: str, 
    results: Dict, 
    api_key: str
) -> Optional[str]:
    """Generate AI-powered recommendations using GPT-4o-mini"""
    
    # Check rate limit
    if cfg.RATE_LIMIT_ENABLED:
        limiter = get_rate_limiter()
        status = limiter.is_allowed()
        
        if not status.is_allowed:
            error_msg = f"Rate limit exceeded. Reset in {int(status.reset_in_seconds)} seconds."
            logger.warning(f"Rate limit hit: {status.requests_in_window}/{status.max_requests}")
            return None
    
    try:
        client = OpenAI(api_key=api_key)
        cost_tracker = get_cost_tracker()
        
        # Format context from search results
        context_parts = []
        for i, meta in enumerate(results['metadatas'][0], 1):
            context_parts.append(f"""
{i}. **{meta['name']}**
   URL: {meta['url']}
   Description: {meta['description'][:300]}...
   Subjects: {meta['subjects']}
   Primary Library: {meta['primary_library']}
   {f"Access Info: {meta['more_info'][:200]}" if meta['more_info'] else ""}
""")
        
        context = "\n".join(context_parts)
        
        # System prompt with strict guidelines
        system_prompt = """You are the Vanderbilt University Library Database Assistant.

Your role is to help students, faculty, and researchers discover the most relevant databases from Vanderbilt's collection.

STRICT RULES:
1. ONLY recommend databases from the provided list
2. NEVER invent or hallucinate database names
3. ALWAYS include the exact database name and URL
4. Explain WHY each database is relevant to the user's query
5. Mention any access requirements or registration needs
6. Be concise but helpful (3-4 sentences per database)
7. Rank recommendations by relevance

Format your response as:
1. Brief intro sentence
2. List of 3-5 recommended databases with explanations
3. Any additional helpful notes about access or usage"""

        log_event("api_call_started", query=query, model=cfg.LLM_MODEL)
        
        response = client.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Research Query: {query}\n\nAvailable Databases:\n{context}\n\nProvide relevant database recommendations:"}
            ],
            temperature=cfg.LLM_TEMPERATURE,
            max_tokens=cfg.LLM_MAX_TOKENS,
            timeout=cfg.LLM_TIMEOUT
        )
        
        # Track cost
        cost = cost_tracker.calculate_cost(
            model=cfg.LLM_MODEL,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )
        cost_tracker.log_cost(cost)
        
        log_event(
            "api_call_completed",
            model=cfg.LLM_MODEL,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=f"${cost.cost_usd:.4f}"
        )
        
        return response.choices[0].message.content
    
    except RateLimitError as e:
        error_msg = "OpenAI API rate limit hit. Try again in 60 seconds."
        logger.error(f"Rate limit error: {e}")
        return None
    
    except APIConnectionError as e:
        error_msg = "Network error connecting to OpenAI. Check your connection and API key."
        logger.error(f"API connection error: {e}")
        return None
    
    except Exception as e:
        logger.exception(f"AI response error: {e}")
        log_event("ai_response_error", query=query, error=str(e))
        return None

# ============================================
# LOAD DATA
# ============================================

try:
    df, csv_file = load_databases()
    
    if df.empty:
        st.error("❌ Failed to load database. Please contact library support.")
        st.stop()
    
    collection = init_vector_search(df)
    if collection is None:
        st.error("❌ Failed to initialize ChromaDB semantic search.")
        st.stop()

except Exception as e:
    logger.exception(f"Fatal initialization error: {e}")
    st.error(f"❌ Application error: {str(e)}")
    st.stop()

# ============================================
# HEADER
# ============================================

col1, col2 = st.columns([4, 1])

with col1:
    st.title("📚 Vanderbilt Database Assistant")
    st.markdown("_Your AI-powered guide to finding the perfect research database_")

with col2:
    st.markdown(f"""
    <div style='text-align: center; padding-top: 20px;'>
        <div class='update-badge'>
            Last Updated<br>{get_last_update_date()}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://cdn.vanderbilt.edu/vu-wp0/wp-content/uploads/sites/59/2019/04/17094551/vu-logo.png", width=200)
    
    st.markdown("### 📊 Database Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h2>{len(df)}</h2>
            <p>Total<br>Databases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_subjects = len(df['Subjects'].str.split(',').explode().str.strip().unique())
        st.markdown(f"""
        <div class='metric-card'>
            <h2>{unique_subjects}</h2>
            <p>Subject<br>Areas</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Subject filter
    st.markdown("### 🔍 Filter by Subject")
    all_subjects = df['Subjects'].str.split(',').explode().str.strip().unique()
    all_subjects = sorted([s for s in all_subjects if s and s != 'nan'])
    
    selected_subjects = st.multiselect(
        "Select subjects:",
        options=all_subjects,
        help="Filter databases by subject area"
    )
    
    st.markdown("---")
    
    # Example searches
    st.markdown("### 💡 Example Questions")
    st.caption("Click to try:")
    
    examples = [
        "🤖 AI tools for legal research",
        "📊 Business analytics and market data",
        "🧬 Biomedical research databases",
        "📰 Historical newspapers",
        "🎵 Music streaming and scores",
        "📚 Literature and humanities",
        "⚖️ Legal case law and statutes",
        "💊 Drug information and pharmacology",
        "🎬 Documentary films",
        "📈 Economic and financial data"
    ]
    
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.query = ex.split(' ', 1)[1]
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("""
    This AI assistant helps you discover relevant databases from Vanderbilt's 
    collection. It uses semantic search and GPT-4 to provide personalized recommendations.
    """)
    
    # Cost & performance stats (admin info)
    with st.expander("📊 Performance Metrics"):
        cost_tracker = get_cost_tracker()
        stats = cost_tracker.get_stats()
        st.metric("Total API Cost", stats['total_cost_usd'])
        st.metric("API Calls", stats['call_count'])
        st.metric("Avg Cost/Call", stats['avg_cost_per_call'])
    
    st.markdown("---")
    st.caption("📧 Questions? [Contact Library Support](mailto:library@vanderbilt.edu)")

# ============================================
# MAIN SEARCH INTERFACE
# ============================================

st.markdown("### 🔎 What are you researching?")

query = st.text_input(
    "Enter your research topic or question:",
    value=st.session_state.query,
    placeholder="e.g., 'legal analytics tools' or 'AI for business research'",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("Clear", use_container_width=True)
    if clear_button:
        st.session_state.query = ""
        st.rerun()

# ============================================
# SEARCH RESULTS
# ============================================

if (search_button or query) and query.strip():
    
    # Add to search history
    if query not in st.session_state.search_history:
        if len(st.session_state.search_history) >= cfg.MAX_SESSION_HISTORY:
            st.session_state.search_history.pop(0)
        st.session_state.search_history.append(query)
    
    with st.spinner("🔍 Searching databases..."):
        
        # Perform search
        results = search_databases(query, collection, cfg.SEARCH_TOP_K)
        
        if results and results['metadatas']:
            
            # Get API key
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except KeyError:
                st.error("⚠️ OpenAI API key not configured")
                logger.error("OPENAI_API_KEY not found in secrets")
                st.stop()
            
            # Generate AI response
            ai_response = generate_ai_response(query, results, api_key)
            
            if ai_response:
                st.markdown("---")
                st.markdown("## 🎯 Recommendations")
                st.markdown(ai_response)
                
                # Display detailed database cards
                st.markdown("---")
                st.markdown("## 📚 Database Details")
                
                for i, meta in enumerate(results['metadatas'][0]):
                    with st.expander(f"**{i+1}. {meta['name']}**", expanded=(i==0)):
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**📖 Description:**")
                            st.write(meta['description'])
                            
                            if meta['subjects']:
                                st.markdown(f"**🏷️ Subjects:** {meta['subjects']}")
                            
                            if meta['primary_library']:
                                st.markdown(f"**🏛️ Primary Library:** {meta['primary_library']}")
                            
                            if meta['alt_names']:
                                st.caption(f"_Also known as: {meta['alt_names']}_")
                        
                        with col2:
                            full_url = meta['url']
                            if meta['friendly_url']:
                                full_url = f"https://researchguides.library.vanderbilt.edu/{meta['friendly_url']}"
                            
                            st.markdown(f"### [🔗 Access Database]({full_url})")
                            
                            if meta['more_info'] and len(meta['more_info']) > 10:
                                st.info(f"ℹ️ {meta['more_info'][:200]}")
                
                # Feedback section
                st.markdown("---")
                st.markdown("### 💬 Was this helpful?")
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button("👍 Yes", use_container_width=True):
                        st.session_state.positive_feedback.append({
                            'query': query,
                            'timestamp': datetime.now().isoformat()
                        })
                        log_event("feedback_positive", query=query)
                        st.success("Thank you for your feedback!")
                
                with col2:
                    if st.button("👎 No", use_container_width=True):
                        st.session_state.negative_feedback.append({
                            'query': query,
                            'timestamp': datetime.now().isoformat()
                        })
                        log_event("feedback_negative", query=query)
                        st.warning("We'll work on improving results!")
            
            else:
                st.error("Unable to generate recommendations. Please try again.")
        
        else:
            st.warning("No results found. Please try a different search term.")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Vanderbilt University Library © 2025</p>
    <p>
        <a href='https://library.vanderbilt.edu' target='_blank'>Library Home</a> | 
        <a href='https://researchguides.library.vanderbilt.edu/az/databases' target='_blank'>All Databases A-Z</a> | 
        <a href='mailto:library@vanderbilt.edu'>Contact Support</a>
    </p>
    <p style='font-size: 12px; margin-top: 10px;'>
        Powered by OpenAI GPT-4o-mini | Data updated weekly
    </p>
</div>
""", unsafe_allow_html=True)
