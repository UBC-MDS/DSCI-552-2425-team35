# test_model_eval.py
# author: Marek Boulerice
# date: 2024-12-15
# Attribution: This code was adapted from Tiffany Timbers

import pytest
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_eval import eval_model

# Test data setup
score_dummy = pd.DataFrame({
    "Metric": ["F1", "Recall", "Accuracy"],
    "Train": [0.6, 0.7, 0.8],
    "Test": [0.5, 0.55, 0.5555],   
})