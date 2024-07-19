import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from flask import Flask, request, jsonify, render_template

# Load the dataset
df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Movies%20Recommendation.csv')
print('Dataset head:\n', df.head())
df.info()
print('Dataset shape:', df.shape)
print('Dataset columns:', df.columns)

# Prepare the features
df_features = df[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']].fillna('')
print('Feature shape:', df_features.shape)
print('Features:\n', df_features.head())

# Combine features into a single string for each movie
x = df_features['Movie_Genre'] + ' ' + df_features['Movie_Keywords'] + ' ' + df_features['Movie_Tagline'] + ' ' + \
    df_features['Movie_Cast'] + ' ' + df_features['Movie_Director']
print('Combined features:\n', x.head())

# Vectorize the combined features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
x = tfidf.fit_transform(x)
print('TF-IDF matrix shape:', x.shape)

# Calculate the cosine similarity score
Similarity_Score = cosine_similarity(x)
print('Similarity score shape:', Similarity_Score.shape)


# Movie recommendation function
def get_movie_recommendations(movie_title, df, similarity_score, top_n=10):
    # Get the list of all movie titles
    all_movie_titles = df['Movie_Title'].tolist()

    # Find the closest match to the input movie title
    close_matches = difflib.get_close_matches(movie_title, all_movie_titles)
    if not close_matches:
        return f"No match found for '{movie_title}'"

    # Get the closest match movie title
    close_match = close_matches[0]
    print(f"Closest match for '{movie_title}': {close_match}")

    # Find the index of the closest match movie
    movie_index = df[df.Movie_Title == close_match].index[0]

    # Get the similarity scores for the closest match movie
    similarity_scores = list(enumerate(similarity_score[movie_index]))

    # Sort the movies based on the similarity scores
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top N recommended movies
    recommended_movies = []
    for i, (index, score) in enumerate(sorted_similar_movies[1:top_n + 1], start=1):
        recommended_movie_title = df.loc[index, 'Movie_Title']
        recommended_movies.append((i, recommended_movie_title, score))

    return recommended_movies


# Flask web application
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['GET'])
def recommend():
    movie_name = request.args.get('movie_name')
    if not movie_name:
        return jsonify({'error': 'No movie name provided'}), 400

    recommendations = get_movie_recommendations(movie_name, df, Similarity_Score, top_n=10)
    if isinstance(recommendations, str):  # If no match found, it's a string message
        return jsonify({'error': recommendations}), 404

    return render_template('recommend.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
