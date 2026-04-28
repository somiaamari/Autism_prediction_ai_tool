"""
Core functionality for the Autism Screening Application.

This module provides:
- ML model prediction capabilities
- Utility functions for data processing
- Resource management
- Report generation
"""

# Import key functions to make them available at package level
from .predictor import predict_probability
from .utils import save_screening_data, get_resources, generate_pdf_report

# Optional: Export specific classes if needed
from .predictor import AutismPredictor

# Package metadata
__version__ = "1.0.0"
__author__ = "Autism Screening Development Team"

# Define what gets imported with "from core import *"
__all__ = [
    "predict_probability",
    "save_screening_data", 
    "get_resources",
    "generate_pdf_report",
    "AutismPredictor"
]

# Optional initialization code
print(f"✅ Core module loaded (v{__version__})")