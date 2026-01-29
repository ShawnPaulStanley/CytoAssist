# Streamlit Cloud entry point
# This file imports and runs the main app from the backend directory

import sys
import os

# Get the directory where this script is located
root_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root_dir, "backend")

# Add backend directory to path so imports work correctly
sys.path.insert(0, backend_dir)

# Change working directory to backend so model path resolves correctly
os.chdir(backend_dir)

# Set __file__ for the executed script so path resolution works
app_path = os.path.join(backend_dir, "app.py")
exec(compile(open(app_path).read(), app_path, 'exec'), {'__file__': app_path, '__name__': '__main__'})
