# Streamlit Cloud entry point
# This file imports and runs the main app from the backend directory

import sys
import os

# Add backend directory to path so imports work correctly
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, backend_dir)

# Change working directory to backend so model path resolves correctly
os.chdir(backend_dir)

# Import and run the main app
exec(open(os.path.join(backend_dir, "app.py")).read())
