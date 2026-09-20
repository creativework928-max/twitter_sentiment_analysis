# Twitter Sentiment Analysis

## Overview

This project implements an end-to-end sentiment analysis system
for Twitter/X-style social media text.

The system performs:

- Dataset acquisition
- Data validation
- Data cleaning
- Twitter-specific preprocessing
- Exploratory data analysis
- TF-IDF feature engineering
- Machine learning model comparison
- Sentiment classification
- Model evaluation
- Topic-specific sentiment analysis
- Professional visualization
- Model persistence
- REST API deployment
- Interactive Streamlit dashboard
- Automated Word report generation

---

## Dataset

The project uses the Sentiment140 dataset.

The dataset contains approximately 1.6 million Twitter messages
and sentiment labels.

Labels:

- 0 = Negative
- 4 = Positive

The labels were generated using distant supervision and therefore
should be considered noisy.

---

## Technologies

Python

Libraries:

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- WordCloud
- Joblib
- FastAPI
- Streamlit
- Plotly
- Python-docx

---

## Project Pipeline

```text
Raw Twitter Dataset
        |
        v
Data Validation
        |
        v
Cleaning & Preprocessing
        |
        v
Exploratory Data Analysis
        |
        v
TF-IDF Feature Engineering
        |
        v
Multiple ML Models
        |
        v
Model Evaluation
        |
        v
Best Model Selection
        |
        +----------------+
        |                |
        v                v
Topic Analysis       Prediction API
        |                |
        v                v
Visualizations      Streamlit
        |
        v
Final Report