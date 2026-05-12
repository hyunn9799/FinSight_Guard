"""Application configuration."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"
