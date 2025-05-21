from surya.scripts.run_streamlit_app import streamlit_app_cli
import sys
import os
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    streamlit_app_cli()