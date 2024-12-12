import pandas as pd
import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from scipy.stats import spearmanr
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Section 1: Data Generation

# Generate Fake Data
def generate_fake_data(platform, num_samples):
fake_data = []
for _ in range(num_samples):
fake_data.append({
"platform": platform,
"text": f"This is a sample post about {random.choice(['topic one', 'topic two', 'topic three', 'topic four', 'topic five'])}.",
"created_at": pd.Timestamp.now() - pd.to_timedelta(random.randint(1, 1000), unit='h'),
"user_id": random.randint(1000, 9999)
})
return pd.DataFrame(fake_data)

# Combine fake data from multiple platforms
def create_fake_dataset():
platforms = ["Twitter", "Facebook", "Instagram", "Bluesky", "Mastodon"]
data_frames = [generate_fake_data(platform, 100) for platform in platforms]
return pd.concat(data_frames, ignore_index=True)

# Generate the dataset and print the first few rows
data = create_fake_dataset()
print("Generated Dataset (First 5 Rows):")
print(data.head())

# Generate Fake Survey Data
def generate_survey_data(num_responses=100):
survey_data = []
for _ in range(num_responses):
response = {
"user_id": random.randint(1000, 9999),
"platform": random.choice(["Twitter", "Facebook", "Instagram", "BlueSky", "Mastodon"]),
"civility_score": random.uniform(1, 5),
"usability_score": random.uniform(1, 5),
"echo_chamber_perception": random.choice([True, False]),
"intelligence": random.uniform(80, 160) # IQ-like metric
}
survey_data.append(response)
return pd.DataFrame(survey_data)

# Generate the dataset and print the first few rows
survey_data = generate_survey_data()
print("Generated Dataset (First 5 Rows):")
print(survey_data.head())

# Section 2: Data Cleaning and Preprocessing

# Clean and preprocess text
def clean_text(text):
text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
text = re.sub(r"@\w+|\#\w+", "", text)
text = re.sub(r"[^A-Za-z\s]", "", text)
text = text.lower()
return text

def preprocess_text(text):
stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text)
return " ".join([word for word in tokens if word not in stop_words])

# Clean and preprocess the data
data["clean_text"] = data["text"].apply(clean_text).apply(preprocess_text)
print("\nCleaned and Preprocessed Text (First 5 Rows):")
print(data[["text", "clean_text"]].head())

# Section 3: Sentiment Generation

# Define functions to generate fake civility and intelligence scores
def generate_fake_civility():
"""Generate a fake civility score between 1 and 5."""
return round(random.uniform(1, 5), 2)

def generate_fake_intelligence():
"""Generate a fake intelligence score between 1 and 5."""
return round(random.uniform(1, 5), 2)

# Apply the functions to generate scores
data["civility_score"] = data["clean_text"].apply(lambda x: generate_fake_civility())
data["intelligence_score"] = data["clean_text"].apply(lambda x: generate_fake_intelligence())

# Display a preview of the updated DataFrame
print("\nUpdated Data with Fake Civility and Intelligence Scores (First 5 Rows):")
print(data[["clean_text", "civility_score", "intelligence_score"]].head())

# Section 4: Clustering and Echo Chamber Detection

# Detect Echo Chambers/Clustering
def detect_echo_chambers(texts, num_clusters=3):
#Detect echo chambers by clustering similar content.
#Parameters:
#- texts: List of text data to cluster.
#- num_clusters: Number of clusters to form.
#Returns:
#- labels: Cluster labels for each text entry.

# Step 1: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(texts)
# Step 2: Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(X)
return labels

# Apply clustering for echo chamber detection
data["cluster"] = detect_echo_chambers(data["clean_text"])
print("\nClustering Results (First 5 Rows):")
print(data[["clean_text", "cluster"]].head())

# Section 5: Algorithm Simulation

# Simulate content recommendation algorithms
# Define a function to simulate the recommendation algorithm
def simulate_recommendation_algorithm(data, num_recommendations=10):    
    #Simulate a recommendation algorithm by randomly selecting posts.
    return data.sample(num_recommendations)

# Define a function to determine if a post is controversial
def is_controversial(row, threshold=1.0):
    #Determine if a post is controversial based on the variance in civility scores.
    return row['civility_score'] > threshold

# Simulate the recommendation algorithm
recommended_posts = simulate_recommendation_algorithm(data, num_recommendations=50)

# Analyze the frequency of controversial posts in the recommendations
recommended_posts['controversial'] = recommended_posts.apply(is_controversial, axis=1)
controversial_count = recommended_posts['controversial'].sum()
total_recommendations = len(recommended_posts)

# Calculate the frequency of controversial posts
controversial_frequency = controversial_count / total_recommendations

# Display the results
print(f"Total Recommendations: {total_recommendations}")
print(f"Controversial Posts: {controversial_count}")
print(f"Frequency of Controversial Posts: {controversial_frequency:.2f}")

# Visualization of the results
plt.figure(figsize=(10, 6))
sns.countplot(x='controversial', data=recommended_posts)
plt.title('Frequency of Controversial Posts in Recommendations')
plt.xlabel('Controversial')
plt.ylabel('Count')
plt.show()

# Section 6: Database Integration

# Database setup
def setup_database():
    conn = sqlite3.connect("social_media_analysis.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT,
            text TEXT,
            created_at TEXT,
            user_id TEXT,
            civility_score REAL,
            intelligence_score REAL,
            cluster INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS survey (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            platform TEXT,
            civility_score REAL,
            usability_score REAL,
            echo_chamber_perception BOOLEAN,
            intelligence_score REAL
        )
    """)
    conn.commit()
    return conn

# Insert data into the database
def store_data(conn, dataframe):
    dataframe.to_sql("posts", conn, if_exists="append", index=False)

# Section 9: Machine Learning: Relationships and Analysis

# Civility and Intelligence
def analyze_relationships(data):
    """
    Use Linear Regression to analyze relationships.
    """
    # Civility and Intelligence Relationship
    X = data[["civility_score"]]
    y = data["intelligence_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    return model

relationship_model = analyze_relationships(data)
