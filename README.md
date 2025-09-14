# Mineral Water Clustering Analysis

## Overview
This project performs clustering analysis on mineral water data to identify patterns and groupings based on their chemical compositions. The analysis uses hierarchical clustering methods to categorize different types of mineral water based on their mineral content.

## Dataset
The dataset contains information about various mineral waters including their:
- Chemical composition
- Mineral content
- pH levels
- Other relevant properties

The data is stored in `data/mineral-waters.xls`.

## Methods
The project implements two main clustering approaches:
1. Ward's method hierarchical clustering
2. Complete linkage hierarchical clustering

The analysis includes:
- Data preprocessing and normalization
- Hierarchical clustering analysis
- Dendrogram visualization
- Cluster interpretation

## Dependencies
- Python 3.11
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xlrd
- openpyxl

## Installation
```bash
# Clone the repository
git clone https://github.com/cruchau/mineral-water-clustering.git

# Navigate to the project directory
cd mineral-water-clustering

# Install dependencies using Poetry
poetry install
