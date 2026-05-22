"""
integration/config.py
---------------------
Centralized Configuration Management
- Paths (BASE_DIR, DATA_DIR, LOGS_DIR, etc.)
- Environment variables (LLM, LangFuse, AWS)
- Service ports (Orchestrator, MCP servers, FastAPI)
- Model configurations
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════
# LOAD ENVIRONMENT VARIABLES
# ══════════════════════════════════════════════════════════════

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / "langfuse_api", override=False)

# ══════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = BASE_DIR / "agents"
RAG_MCP_DIR = BASE_DIR / "rag_mcp"
REPORTING_MCP_DIR = BASE_DIR / "reporting_agent_mcp"
MCP_SERVERS_DIR = BASE_DIR / "mcp_servers"

# Data directories
DATA_DIR = BASE_DIR / "data"
INCOMING_DIR = DATA_DIR / "incoming"
ERP_DIR = DATA_DIR / "ERP_mockdata"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
FAISS_INDEX_PATH = VECTOR_DB_DIR / "invoice_index.faiss"
CHUNKS_JSON_PATH = VECTOR_DB_DIR / "chunks.json"

# Config directories
CONFIG_DIR = BASE_DIR / "config"
RULES_PATH = CONFIG_DIR / "rules.yaml"
PERSONAS_PATH = CONFIG_DIR / "agent_personas.yaml"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# UI directories
UI_DIR = BASE_DIR / "ui"

# Schemas
SCHEMAS_DIR = BASE_DIR / "schemas"

# ══════════════════════════════════════════════════════════════
# ENSURE DIRECTORIES EXIST
# ══════════════════════════════════════════════════════════════

def init_directories():
    """Create all required directories if they don't exist."""
    for dir_path in [
        INCOMING_DIR,
        VECTOR_DB_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# AWS & LLM CONFIGURATION
# ══════════════════════════════════════════════════════════════

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# LiteLLM Configuration (centralized LLM gateway)
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://localhost:8000")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "bedrock/cohere.command-r-plus-v1:0")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))

# Embedding Model (for RAG - uses local SentenceTransformer)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ══════════════════════════════════════════════════════════════
# LANGFUSE CONFIGURATION (Observability)
# ══════════════════════════════════════════════════════════════

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# ══════════════════════════════════════════════════════════════
# SERVICE PORTS (for background processes)
# ══════════════════════════════════════════════════════════════

ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8001"))
REPORTING_MCP_PORT = int(os.getenv("REPORTING_MCP_PORT", "8004"))
EXTRACTION_MCP_PORT = int(os.getenv("EXTRACTION_MCP_PORT", "8005"))
TRANSLATION_MCP_PORT = int(os.getenv("TRANSLATION_MCP_PORT", "8006"))
RAG_MCP_PORT = int(os.getenv("RAG_MCP_PORT", "8007"))
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8002"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))


# =====================================================
# AWS BEDROCK CONFIG
# =====================================================

BEDROCK_MODEL = (
    "anthropic.claude-3-haiku-20240307-v1:0"
)

BEDROCK_REGION = (
    "us-east-1"
)

USE_BEDROCK = os.getenv("INVOICE_USE_BEDROCK", "0").lower() in {
    "1",
    "true",
    "yes",
}

# =====================================================
# RAG CONFIG
# =====================================================

RAG_TOP_K = 5

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

# =====================================================
# TRANSLATION CONFIG
# =====================================================

MIN_TRANSLATION_CONFIDENCE = 0.7

# =====================================================
# DEBUG
# =====================================================

DEBUG_MODE = False

# ══════════════════════════════════════════════════════════════
# VALIDATION & RAG CONFIGURATION
# ══════════════════════════════════════════════════════════════

# RAG Pipeline settings
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "300"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))

# Validation settings
MIN_TRANSLATION_CONFIDENCE = float(os.getenv("MIN_TRANSLATION_CONFIDENCE", "0.85"))
AUTO_APPROVE_CONFIDENCE = float(os.getenv("AUTO_APPROVE_CONFIDENCE", "0.95"))

# File monitoring
MONITOR_SETTLE_SECONDS = float(os.getenv("MONITOR_SETTLE_SECONDS", "2.0"))
MONITOR_WATCH_TIMEOUT = int(os.getenv("MONITOR_WATCH_TIMEOUT", "300"))

# ══════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "invoice_auditor.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_config_summary() -> str:
    """Return a summary of current configuration."""
    return f"""
╔════════════════════════════════════════════════════════╗
║           AI INVOICE AUDITOR CONFIGURATION            ║
╚════════════════════════════════════════════════════════╝

📁 PATHS:
   Base Directory     : {BASE_DIR}
   Data Directory     : {DATA_DIR}
   Logs Directory     : {LOGS_DIR}
   Reports Directory  : {REPORTS_DIR}
   Vector DB          : {VECTOR_DB_DIR}

🤖 LLM:
   Model              : {LITELLM_MODEL}
   Gateway            : {LITELLM_API_BASE}
   Temperature        : {LLM_TEMPERATURE}

🔍 RAG:
   Chunk Size         : {RAG_CHUNK_SIZE}
   Top K Results      : {RAG_TOP_K}
   Embedding Model    : {EMBEDDING_MODEL_NAME}

🔐 Observability:
   LangFuse Public Key: {LANGFUSE_PUBLIC_KEY[:20]}...
   LangFuse URL       : {LANGFUSE_BASE_URL}

⚙️ SERVICES:
   Orchestrator       : {ORCHESTRATOR_PORT}
   FastAPI            : {FASTAPI_PORT}
   Streamlit          : {STREAMLIT_PORT}
   Reporting MCP      : {REPORTING_MCP_PORT}
   Extraction MCP     : {EXTRACTION_MCP_PORT}
   Translation MCP    : {TRANSLATION_MCP_PORT}
   RAG MCP            : {RAG_MCP_PORT}

"""


if __name__ == "__main__":
    init_directories()
    print(get_config_summary())
