#!/usr/bin/env python3
"""Unified Launcher for Tehri Hydroelectric Operational System & Gov Portal.

Usage:
    python3 run.py             # Launches National Hydroelectric Operations Portal in browser
    python3 run.py --streamlit # Launches Streamlit Operator Dashboard (using .venv)
    python3 run.py --pipeline  # Runs analytics, training, and report generation
"""

import argparse
import http.server
import os
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PORTAL_DIR = BASE_DIR / "portal"
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
VENV_STREAMLIT = BASE_DIR / ".venv" / "bin" / "streamlit"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PORTAL_DIR), **kwargs)

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[NHOP-Portal] {args[0]} - {args[1]}\n")


def run_portal(port: int = 8000):
    """Start web server for the Government Web Portal and auto-open browser."""
    print("=" * 70)
    print("  NATIONAL HYDROELECTRIC OPERATIONS & GRID DISPATCH PORTAL (NHOP)")
    print("  Government of India • Ministry of Power • CEA • THDC India Ltd")
    print("=" * 70)
    print(f"\n[INFO] Starting National Portal at: http://localhost:{port}")
    print("[INFO] Opening browser automatically...")

    url = f"http://localhost:{port}"
    webbrowser.open(url)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"[INFO] Portal is LIVE and serving! Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down portal server. Goodbye!")


def run_streamlit():
    """Launch Streamlit dashboard via local virtual environment."""
    if not VENV_STREAMLIT.exists():
        print(f"Error: {VENV_STREAMLIT} not found. Please ensure virtual environment is created.")
        sys.exit(1)

    print("Launching Streamlit Operator Console...")
    cmd = [str(VENV_STREAMLIT), "run", "app/dashboard.py"]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_pipeline():
    """Run data ingestion and forecasting pipeline."""
    py_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    print("Running Analytics & Machine Learning Pipeline...")
    cmd = [py_exec, "hydro_power_analytics.py"]
    subprocess.run(cmd, cwd=str(BASE_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tehri Hydroelectric Operational System Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Port for the Gov Portal (default: 8000)")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit dashboard instead")
    parser.add_argument("--pipeline", action="store_true", help="Run ML pipeline & generate figures")

    args = parser.parse_args()

    if args.pipeline:
        run_pipeline()
    elif args.streamlit:
        run_streamlit()
    else:
        run_portal(port=args.port)
