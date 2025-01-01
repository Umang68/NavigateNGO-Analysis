import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline



from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

df = pd.read_csv('NGO Response - Form responses 1.csv')
df.shape

df.head(2)

df.describe()


df.columns


print(df.info)


def check_data_cleanliness(df):
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("There are missing values in the dataset:")
        print(missing_values)
    else:
        print("No missing values found in the dataset.")
    outlier_threshold = 3
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = (np.abs(df[numeric_columns] - df[numeric_columns].mean()) >
                outlier_threshold * df[numeric_columns].std())
    if outliers.any().any():
        print("There are outliers in the dataset:")
        print(outliers.sum())
    else:
        print("No outliers found in the dataset.")
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        print("There are duplicate rows in the dataset:", duplicate_rows)
    else:
        print("No duplicate rows found in the dataset.")
check_data_cleanliness(df)


df
#Sentiment Analysis of NGO Engagement

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('NGO Response - Form responses 1.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on the 'NGO Engagement' column
def analyze_sentiment(text):
    sentiment_score = sid.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['NGO Engagement'].apply(analyze_sentiment)

# Count the number of occurrences for each sentiment category
sentiment_counts = df['Sentiment'].value_counts()

# Plotting the sentiment distribution
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['black', 'red', 'blue'])
plt.title('Sentiment Analysis of NGO Engagement')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Print the sentiment analysis results
print(df[['NGO Engagement', 'Sentiment']])


#Logistic Regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Drop rows with missing values
df.dropna(inplace=True)

# Check if data is empty or contains NaN values
if df.empty:
    raise ValueError("DataFrame is empty")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical variables into numerical labels
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Define features (X) and target variable (y)
X = df.drop(columns=['Personal Stories'])  # Replace 'Target_Variable' with your target variable column name
y = df['Personal Stories']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=70)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Check if y_pred and y_test are empty or contain NaN values
if y_pred.size == 0 or y_test.size == 0:
    raise ValueError("Predictions or true labels are empty")

# Calculate class frequencies for predictions and true labels
pred_counts = [list(y_pred).count(i) for i in range(11)]
true_counts = [list(y_test).count(i) for i in range(11)]

# Visualize the output using a bar graph
plt.figure(figsize=(6, 4))
classes = list(range(11))
bar_width = 0.35
index = range(11)
plt.bar(index, pred_counts, bar_width, color='pink', label='Predictions')
plt.bar([i + bar_width for i in index], true_counts, bar_width, color='black', label='True Labels')
plt.xlabel('Responses')
plt.ylabel('Frequency')
plt.title('Bar Graph of Predictions vs True Labels')
plt.xticks([i + bar_width/2 for i in index], classes)
plt.legend()
plt.tight_layout()
plt.show()


#Kmeans Clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Drop rows with missing values
df.dropna(inplace=True)

# Check if data is empty or contains NaN values
if df.empty:
    raise ValueError("DataFrame is empty")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical variables into numerical labels
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Define features (X) and target variable (y)
X = df.drop(columns=['Post Frequency'])  # Replace 'Target_Variable' with your target variable column name
y = df['Post Frequency']

# Split the data into train and test sets (not required for clustering, but kept for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

# Initialize and train the KMeans clustering model
model = KMeans(n_clusters=3, random_state=0)  # Assuming 3 clusters, adjust as needed
model.fit(X)

# Make predictions on the cluster labels
y_pred = model.labels_

# Note: KMeans is an unsupervised algorithm, so there's no accuracy score, classification report, or confusion matrix

# Plot the cluster centroids (optional)
plt.figure(figsize=(6, 4))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='viridis')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='*', s=300, c='red', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()


#Decision tree Classifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Drop rows with missing values
df.dropna(inplace=True)

# Check if data is empty or contains NaN values
if df.empty:
    raise ValueError("DataFrame is empty")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical variables into numerical labels
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Define features (X) and target variable (y)
X = df.drop(columns=['Engaging Content'])  # Replace 'Target_Variable' with your target variable column name
y = df['Engaging Content']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# Initialize and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Check if y_pred and y_test are empty or contain NaN values
if y_pred.size == 0 or y_test.size == 0:
    raise ValueError("Predictions or true labels are empty")

# Visualize the output using a scatter plot
plt.figure(figsize=(6, 4))
plt.scatter(range(len(y_pred)), y_pred, label='Predictions', color='blue', marker='o')
plt.scatter(range(len(y_test)), y_test, label='True Labels', color='red', marker='x')
plt.xlabel('Responses')
plt.ylabel('Frequency')
plt.title('Scatter Plot of Predictions vs True Labels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Sentiment Analysis of NGO Outreach Satisfaction
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('NGO Response - Form responses 1.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on the 'NGO Engagement' column
def analyze_sentiment(text):
    sentiment_score = sid.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Outreach Satisfaction'].apply(analyze_sentiment)

# Count the number of occurrences for each sentiment category
sentiment_counts = df['Sentiment'].value_counts()

# Plotting the sentiment distribution as a pie chart
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green','red','orange'])
plt.title('Sentiment Analysis of NGO Outreach Satisfaction')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Print the sentiment analysis results
print(df[['Outreach Satisfaction', 'Sentiment']])
