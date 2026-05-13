"""Application configuration."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
ENABLE_LLM_SUPERVISOR = os.getenv("ENABLE_LLM_SUPERVISOR", "false").strip().lower() == "true"
LLM_SUPERVISOR_MODEL = os.getenv("LLM_SUPERVISOR_MODEL", LLM_MODEL)
