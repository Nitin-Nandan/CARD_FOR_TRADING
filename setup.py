#!/usr/bin/env python3
"""
Setup Script: Initialize Project Structure

Creates all necessary directories and placeholder files.

Usage:
    python setup.py
"""

from pathlib import Path

print("="*60)
print("CARD STOCK PREDICTION - PROJECT SETUP")
print("="*60)

# Project root
PROJECT_ROOT = Path(".")

# Define directory structure
directories = [
    # Data directories
    "data/raw",
    "data/market_indices",
    "data/processed",
    "data/combined",
    "data/windows",
    
    # Model directories
    "models/checkpoints",
    
    # Loss directories
    "losses",
    
    # Script directories
    "scripts",
    
    # Results directories
    "results/training_curves",
    "results/predictions",
    "results/metrics",
    
    # Logs
    "logs",
    
    # Notebooks
    "notebooks",
    
    # Tests
    "tests",
]

print("\nCreating directory structure...")

for directory in directories:
    dir_path = PROJECT_ROOT / directory
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep to preserve directory in git
    gitkeep = dir_path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
    
    print(f"  ✅ {directory}")

# Create __init__.py files
init_files = [
    "models/__init__.py",
    "losses/__init__.py",
    "tests/__init__.py",
]

print("\nCreating __init__.py files...")

for init_file in init_files:
    init_path = PROJECT_ROOT / init_file
    if not init_path.exists():
        init_path.write_text('"""Package initialization"""\n')
        print(f"  ✅ {init_file}")

# Create .env template if doesn't exist
env_template = PROJECT_ROOT / ".env.template"
if not env_template.exists():
    env_template.write_text("""# Fyers API Credentials
CLIENT_ID=your_client_id_here
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN=generate_using_00_generate_token.py

# Optional: Custom paths
DATA_DIR=./data
MODEL_DIR=./models
RESULTS_DIR=./results
""")
    print("\n  ✅ .env.template created")
    print("     → Copy to .env and fill in your credentials")

# Check if .env exists
env_file = PROJECT_ROOT / ".env"
if not env_file.exists():
    print("\n⚠️  .env file not found!")
    print("   Please create .env from .env.template")
else:
    print("\n  ✅ .env file exists")

print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)

print("\nNext steps:")
print("  1. Install dependencies:")
print("     pip install -r requirements.txt")
print("")
print("  2. Setup Fyers credentials in .env")
print("")
print("  3. Generate access token:")
print("     python scripts/00_generate_token.py")
print("")
print("  4. Download Nifty 50 data:")
print("     python scripts/01_download_nifty50.py")
print("")
print("="*60)
