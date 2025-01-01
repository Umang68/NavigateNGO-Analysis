The objective of this project is to analyze NGO engagement and outreach satisfaction using survey data. It aims to perform sentiment analysis on textual responses, apply machine learning models to predict key outcomes, and visualize the results to uncover insights that can help NGOs improve their engagement strategies and participant satisfaction.

File Descriptions
NGO Response - Form responses 1.csv: This is the dataset containing responses from the survey.
Sentiment_Analysis.py: Python script to perform sentiment analysis on the "NGO Engagement" and "Outreach Satisfaction" columns.
Machine_Learning_Models.py: Python script for training and evaluating machine learning models including Logistic Regression, KMeans Clustering, and Decision Tree Classifier.
Data Cleaning and Preprocessing
The project starts by checking for missing values, duplicates, and outliers in the dataset. This ensures that the data is clean before running any analysis or machine learning models.

Sentiment Analysis
The "NGO Engagement" and "Outreach Satisfaction" columns are analyzed for sentiment using the VADER sentiment analysis tool.
The results are visualized using bar charts and pie charts for a better understanding of the distribution of sentiment.
Machine Learning Models
Logistic Regression: Trains a classifier to predict the "Personal Stories" column based on other features in the dataset.
KMeans Clustering: Clusters the data into 3 groups and visualizes the clusters using scatter plots.
Decision Tree Classifier: Trains a classifier to predict the "Engaging Content" column based on other features.
Visualizations
Various visualizations are used throughout the project:

Bar charts for sentiment distribution.
Pie chart for sentiment analysis results of outreach satisfaction.
Scatter plots to visualize model predictions vs true labels.
