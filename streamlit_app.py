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
    is_query_too_vague,
    rank_databases_from_spreadsheet,
    verify_prd_candidates,
    build_query_matched_explanation,
    RESEARCH_GUIDES_FALLBACK_URL,
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
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = None

# ============================================
# DATA LOADING & VALIDATION
# ============================================

def normalize_database_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize varying Vanderbilt export schemas into canonical column names."""
    normalized = df.copy()
    normalized.columns = [str(c).strip() for c in normalized.columns]

    # Prefer explicit header names from full export files.
    explicit_mapping = {
        "Alt. Names / Keywords": "Alt_Names",
        "Friendly URL": "Friendly_URL",
        "More Info": "More_Info",
        "Primary Library": "Primary_Library",
        "Last Updated": "Last_Updated",
    }
    normalized = normalized.rename(columns={k: v for k, v in explicit_mapping.items() if k in normalized.columns})

    # Positional fallback for older 12-column exports.
    if len(normalized.columns) >= 12:
        positional_targets = {
            0: "ID",
            1: "Name",
            2: "Description",
            3: "URL",
            4: "Last_Updated",
            5: "Primary_Library",
            6: "Alt_Names",
            8: "Friendly_URL",
            10: "Subjects",
            11: "More_Info",
        }
        for idx, target in positional_targets.items():
            if idx < len(normalized.columns) and target not in normalized.columns:
                normalized = normalized.rename(columns={normalized.columns[idx]: target})

    # Drop duplicate column names (can happen with mixed explicit+positional mappings).
    normalized = normalized.loc[:, ~normalized.columns.duplicated()]
    return normalized


@st.cache_data(ttl=cfg.CSV_CACHE_TTL)
def load_databases() -> Tuple[pd.DataFrame, Optional[str]]:
    """Load and validate the most recent database CSV file"""
    try:
        csv_files = glob.glob(cfg.CSV_GLOB_PATTERN)

        if not csv_files:
            error_msg = f"No database CSV found in {cfg.CSV_GLOB_PATTERN}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        candidates = sorted(csv_files, key=os.path.getmtime, reverse=True)
        best_df: Optional[pd.DataFrame] = None
        best_file: Optional[str] = None
        best_rows = -1

        for candidate in candidates:
            try:
                cdf = pd.read_csv(candidate).fillna("")
                cdf = normalize_database_columns(cdf)

                validation_result = validate_csv(cdf)
                rows = len(cdf)

                if validation_result.is_valid and rows >= cfg.CSV_MIN_ROWS:
                    logger.info(f"Selected CSV: {candidate} ({rows} rows)")
                    log_event("csv_loaded", file=candidate, records=rows)
                    return cdf, candidate

                if validation_result.is_valid and rows > best_rows:
                    best_rows = rows
                    best_df = cdf
                    best_file = candidate
            except Exception as e:
                logger.warning(f"Skipping CSV candidate {candidate}: {e}")

        if best_df is not None and best_file is not None:
            logger.warning(f"Using fallback CSV: {best_file} ({best_rows} rows)")
            log_event("csv_loaded_fallback", file=best_file, records=best_rows)
            return best_df, best_file

        raise ValueError("No valid CSV file found after evaluating candidates")

    except FileNotFoundError as e:
        logger.error(f"CSV not found: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error loading database: {e}")
        raise


def get_last_update_date() -> str:
    """Get timestamp of last database update"""
    try:
        _, selected_csv = load_databases()
        if selected_csv:
            timestamp = os.path.getmtime(selected_csv)
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

        # Reuse existing collection when it already has all documents indexed.
        # On Streamlit Cloud, chroma_data/ persists across server restarts, so
        # skipping re-indexing dramatically reduces cold-start time.
        expected_count = len(_df)
        try:
            collection = client.get_collection("vanderbilt_databases")
            if collection.count() == expected_count:
                logger.info(f"ChromaDB: reusing existing collection ({expected_count} docs)")
                log_event("vector_search_reused", database_count=expected_count)
                return collection
        except Exception:
            pass

        # Collection missing or stale — rebuild it.
        try:
            client.delete_collection("vanderbilt_databases")
        except Exception:
            pass

        collection = client.create_collection(
            name="vanderbilt_databases",
            metadata={"description": "Vanderbilt Database A-Z List"}
        )

        # Build batch lists — single collection.add() is dramatically faster than
        # 890 individual calls (one model inference pass vs. 890 separate passes)
        all_documents: list = []
        all_metadatas: list = []
        all_ids: list = []
        seen_ids: set = set()

        for idx, row in _df.iterrows():
            raw_id = str(row.get('ID', idx))
            # Deduplicate IDs — ChromaDB errors on duplicates
            unique_id = raw_id
            suffix = 1
            while unique_id in seen_ids:
                unique_id = f"{raw_id}_{suffix}"
                suffix += 1
            seen_ids.add(unique_id)

            search_text = (
                f"NAME: {row.get('Name', '')}\n"
                f"DESCRIPTION: {row.get('Description', '')}\n"
                f"SUBJECTS: {row.get('Subjects', '')}\n"
                f"KEYWORDS: {row.get('Alt_Names', '')}\n"
                f"LIBRARY: {row.get('Primary_Library', '')}\n"
                f"INFO: {row.get('More_Info', '')}\n"
            )

            metadata = {
                'id': raw_id,
                'name': str(row.get('Name', 'Unknown'))[:500],
                'url': str(row.get('URL', '')),
                'description': str(row.get('Description', 'No description'))[:1000],
                'subjects': str(row.get('Subjects', ''))[:500],
                'primary_library': str(row.get('Primary_Library', ''))[:200],
                'alt_names': str(row.get('Alt_Names', ''))[:300],
                'more_info': str(row.get('More_Info', ''))[:500],
                'friendly_url': str(row.get('Friendly_URL', ''))
            }

            all_documents.append(search_text)
            all_metadatas.append(metadata)
            all_ids.append(unique_id)

        # Single batch add — one embedding model pass for all documents
        collection.add(
            documents=all_documents,
            metadatas=all_metadatas,
            ids=all_ids
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
# QUERY EXPANSION
# ============================================

def expand_query(query: str, amp_token: str) -> str:
    """Use the LLM to expand a user query with relevant subject terms for better semantic search.
    
    For example, 'Judith Herman' -> 'Judith Herman trauma PTSD psychology psychiatry mental health research'
    """
    try:
        payload = {
            "data": {
                "temperature": 0.3,
                "max_tokens": 150,
                "dataSources": [],
                "messages": [
                    {"role": "system", "content": (
                        "You are a research librarian. Given a user's search query, expand it with "
                        "relevant academic subject areas, disciplines, and keywords that would help "
                        "find the right research databases. If the query is a person's name, identify "
                        "their field of study and add relevant subject terms. If it's already a topic, "
                        "add related disciplines and synonyms.\n\n"
                        "Reply ONLY with the expanded search string — no explanation, no bullet points, "
                        "no formatting. Keep the original query and add 5-10 relevant terms."
                    )},
                    {"role": "user", "content": query}
                ],
                "options": {
                    "ragOnly": False,
                    "skipRag": True,
                    "model": {"id": cfg.LLM_MODEL},
                    "prompt": query
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {amp_token}",
            "Content-Type": "application/json"
        }
        resp = requests.post(
            f"{cfg.AMPLIFY_BASE_URL}/chat",
            json=payload,
            headers=headers,
            timeout=cfg.LLM_TIMEOUT
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("success") and result.get("data"):
            expanded = result["data"].strip()
            log_event("query_expanded", original=query, expanded=expanded)
            return expanded
    except Exception as e:
        logger.warning(f"Query expansion failed, using original: {e}")
    return query

# ============================================
# AI RESPONSE GENERATION
# ============================================

def generate_ai_response(
    query: str, 
    results: Dict, 
    amp_token: str
) -> Tuple[Optional[str], Optional[str]]:
    """Generate AI-powered recommendations using Vanderbilt Amplify API"""
    
    # Check rate limit
    if cfg.RATE_LIMIT_ENABLED:
        limiter = get_rate_limiter()
        status = limiter.is_allowed()
        
        if not status.is_allowed:
            error_msg = f"Rate limit exceeded. Reset in {int(status.reset_in_seconds)} seconds."
            logger.warning(f"Rate limit hit: {status.requests_in_window}/{status.max_requests}")
            return None, error_msg
    
    try:
        # Format context from search results
        context_parts = []
        for i, meta in enumerate(results['metadatas'][0], 1):
            # Only use Friendly URL
            if meta.get('friendly_url') and meta['friendly_url'] not in ('', 'nan'):
                display_url = f"https://researchguides.library.vanderbilt.edu/{meta['friendly_url']}"
                url_line = f"   URL: {display_url}"
            else:
                url_line = ""
            context_parts.append(f"""
{i}. **{meta['name']}**
{url_line}
   Description: {meta['description'][:300]}...
   Subjects: {meta['subjects']}
   Primary Library: {meta['primary_library']}
   {f"Access Info: {meta['more_info'][:200]}" if meta.get('more_info') else ""}
""")
        context = "\n".join(context_parts)

        # Strict Vanderbilt Library system prompt
        system_prompt = """You are the Vanderbilt University Library Database Assistant.

Your role is to help students, faculty, and researchers discover the most relevant databases from Vanderbilt's collection for their research needs.

IMPORTANT REASONING STEPS:
1. First, determine the SUBJECT AREA of the query. If the user mentions a person's name, identify what field(s) that person works in and treat the query as a search for databases in those fields.
2. From the provided database list, select ONLY those whose subjects, descriptions, or content areas genuinely match the research field.
3. Do NOT recommend a database just because its name or description contains a superficial keyword match. Focus on subject-matter relevance.

STRICT RULES:
1. ONLY recommend databases from the provided list — never invent names
2. Use the EXACT database name as listed — do not paraphrase or modify names
3. Rank recommendations by subject relevance to the research field, not by keyword overlap
4. For each database, explain WHY it is uniquely suited to this query — do NOT repeat or paraphrase the database description
5. If fewer than 3 databases are truly relevant, recommend only those — do not pad with poor matches
6. Be concise (1-2 sentences per database insight)
7. Do NOT include URLs — they are displayed alongside the database cards
8. Mention access requirements only if they significantly affect availability (e.g., requires separate registration)

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS (no extra text before or after):

SUMMARY: [1-2 sentence overview identifying the research area and which databases are most useful]

1. [Exact Database Name]
INSIGHT: [1-2 sentences explaining why this database uniquely fits the query — focus on what makes it relevant, not a repeat of its description]

2. [Exact Database Name]
INSIGHT: [1-2 sentences]

(continue for all relevant databases, ranked by relevance)"""

        log_event("api_call_started", query=query, model=cfg.LLM_MODEL, provider="amplify")

        # Amplify /chat payload
        user_content = f"Research Query: {query}\n\nAvailable Databases:\n{context}\n\nProvide relevant database recommendations:"
        payload = {
            "data": {
                "temperature": cfg.LLM_TEMPERATURE,
                "max_tokens": cfg.LLM_MAX_TOKENS,
                "dataSources": [],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "options": {
                    "ragOnly": False,
                    "skipRag": True,
                    "model": {"id": cfg.LLM_MODEL},
                    "prompt": user_content
                }
            }
        }

        headers = {
            "Authorization": f"Bearer {amp_token}",
            "Content-Type": "application/json"
        }

        api_response = requests.post(
            f"{cfg.AMPLIFY_BASE_URL}/chat",
            json=payload,
            headers=headers,
            timeout=cfg.LLM_TIMEOUT
        )
        api_response.raise_for_status()

        result = api_response.json()

        if result.get("success"):
            ai_text = result.get("data")
            log_event("api_call_completed", model=cfg.LLM_MODEL, provider="amplify", success=True)
            return ai_text, None
        else:
            error_msg = result.get("message", "Unknown Amplify error")
            return None, f"Amplify API error: {error_msg}"

    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        logger.error(f"Amplify HTTP {e.response.status_code}: {body}")
        if e.response.status_code == 401:
            return None, "❌ Invalid Amplify token (401). Check AMPLIFY_TOKEN in secrets."
        elif e.response.status_code == 400:
            return None, f"❌ Bad request to Amplify (400): {body}"
        else:
            return None, f"HTTP {e.response.status_code} error from Amplify: {body}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error to Amplify: {e}")
        return None, f"Network error connecting to Amplify API: {str(e)}"
    
    except Exception as e:
        logger.exception(f"Amplify response error: {e}")
        log_event("ai_response_error", query=query, error=str(e), provider="amplify")
        return None, f"Amplify request failed: {str(e)}"


# ============================================
# LOAD DATA
# ============================================

try:
    df, csv_file = load_databases()
    
    if df.empty:
        st.error("❌ Failed to load database. Please contact library support.")
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
    st.markdown(f"""
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <h2 style='color: {cfg.PRIMARY_COLOR}; font-family: Georgia, serif; margin: 0;'>
            VANDERBILT<br>UNIVERSITY LIBRARY
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
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
        unique_subjects = len(df['Subjects'].str.split(r'[,;]', regex=True).explode().str.strip().unique())
        st.markdown(f"""
        <div class='metric-card'>
            <h2>{unique_subjects}</h2>
            <p>Subject<br>Areas</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Subject filter
    st.markdown("### 🔍 Filter by Subject")
    all_subjects = df['Subjects'].str.split(r'[,;]', regex=True).explode().str.strip().unique()
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
    collection. It uses semantic search and the Amplify API to provide personalized recommendations.
    """)
    
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
        st.session_state.last_results = None
        st.session_state.last_query = None
        st.rerun()

# ============================================
# SEARCH RESULTS
# ============================================

if search_button and query.strip():
    
    # Add to search history
    if query not in st.session_state.search_history:
        if len(st.session_state.search_history) >= cfg.MAX_SESSION_HISTORY:
            st.session_state.search_history.pop(0)
        st.session_state.search_history.append(query)
    
    with st.spinner("🔍 Ranking databases from the Databases A-Z spreadsheet..."):
        if is_query_too_vague(query):
            st.session_state.last_results = {
                "clarify": "Could you share one specific subject area or course context so I can rank the top databases accurately?"
            }
            st.session_state.last_query = query
        else:
            ranked = rank_databases_from_spreadsheet(df, query, top_k=cfg.SEARCH_TOP_K)

            if selected_subjects:
                ranked = [
                    item for item in ranked
                    if any(subj.strip() in selected_subjects for subj in str(item.get("subjects", "")).split(","))
                ]

            verified = verify_prd_candidates(ranked, df)

            log_event(
                "prd_verification",
                query=query,
                candidates=len(ranked),
                verified=len(verified),
                discarded=max(0, len(ranked) - len(verified)),
            )

            st.session_state.last_results = {
                "items": verified,
                "candidate_count": len(ranked),
            }
            st.session_state.last_query = query

# Display persisted results
if st.session_state.last_query and st.session_state.last_results:
    results_payload = st.session_state.last_results

    if results_payload.get("clarify"):
        st.markdown("---")
        st.info(results_payload["clarify"])

    items = results_payload.get("items", [])

    # Recommendations summary
    st.markdown("---")
    st.markdown("## 🎯 Recommendations")
    if items:
        st.info(f"Top {len(items)} database match(es) ranked from the Vanderbilt Databases A-Z spreadsheet.")
    else:
        st.info("No strong matches were found in the Databases A-Z list.")

    # Database Details section
    num_shown = len(items)
    st.markdown("---")
    st.markdown(f"## 📚 Database Details")
    st.caption(f"Showing {num_shown} of {len(df)} databases")

    for i, item in enumerate(items):
        with st.expander(f"**{i+1}. {item['name']}**", expanded=(i == 0)):

            col1, col2 = st.columns([3, 1])

            with col1:
                explanation = build_query_matched_explanation(
                    st.session_state.last_query,
                    item.get("description", ""),
                    item.get("subjects", ""),
                )
                st.markdown(f"**🤖 Why this database:** {explanation}")
                st.markdown("---")

                st.markdown(f"**📖 Description:**")
                st.write(item.get("description", ""))

                if item.get("subjects"):
                    st.markdown(f"**🏷️ Subjects:** {item.get('subjects', '')}")

            with col2:
                full_url = item.get("url", RESEARCH_GUIDES_FALLBACK_URL)
                st.markdown(f"### [🔗 Access Database]({full_url})")

                if full_url == RESEARCH_GUIDES_FALLBACK_URL:
                    st.caption(
                        f"Friendly URL is blank for this row. Search for exact name '{item['name']}' on the A-Z page."
                    )

    # Feedback section
    st.markdown("---")
    st.markdown("### 💬 Was this helpful?")
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("👍 Yes", use_container_width=True):
            st.session_state.positive_feedback.append({
                'query': st.session_state.last_query,
                'timestamp': datetime.now().isoformat()
            })
            log_event("feedback_positive", query=st.session_state.last_query)
            st.success("Thank you for your feedback!")

    with col2:
        if st.button("👎 No", use_container_width=True):
            st.session_state.negative_feedback.append({
                'query': st.session_state.last_query,
                'timestamp': datetime.now().isoformat()
            })
            log_event("feedback_negative", query=st.session_state.last_query)
            st.warning("We'll work on improving results!")

elif st.session_state.last_query and not st.session_state.last_results:
    st.markdown("---")
    st.info("🔍 No results found for your search. Try a different keyword or [browse all databases](https://researchguides.library.vanderbilt.edu/az/databases).")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Vanderbilt University Library &copy; 2026</p>
    <p>
        <a href='https://www.library.vanderbilt.edu/' target='_blank'>Library Home</a> | 
        <a href='https://researchguides.library.vanderbilt.edu/az/databases' target='_blank'>All Databases A-Z</a> | 
        <a href='mailto:library@vanderbilt.edu'>Contact Support</a>
    </p>
    <p style='font-size: 12px; margin-top: 10px;'>
        Powered by Vanderbilt Amplify | Data updated weekly
    </p>
</div>
""", unsafe_allow_html=True)
