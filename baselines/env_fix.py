"""
=============================================================================
ENV PATCHER — run this once to set env vars for all baseline scripts,
OR add these two lines at the very top of each script.
=============================================================================
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
