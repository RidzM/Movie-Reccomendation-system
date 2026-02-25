import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_prep_data(filepath):
    """Loads the dataset and handles missing values in text columns."""
    df = pd.read_csv(filepath)
    
    # These are the specific features we'll use for comparison
    features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    
    # Humans often use a loop for filling NaNs in multiple columns
    for col in features:
        df[col] = df[col].fillna('')
        
    # Combine everything into one string for the vectorizer
    df['combined_features'] = df['genres'] + ' ' + df['keywords'] + ' ' + \
                             df['tagline'] + ' ' + df['cast'] + ' ' + \
                             df['director']
    return df

def calculate_similarity(df):
    """Converts text to vectors and calculates the cosine similarity matrix."""
    vectorizer = TfidfVectorizer(stop_words='english') # Added stop_words for higher accuracy
    matrix = vectorizer.fit_transform(df['combined_features'])
    
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix

def get_recommendations(movie_title, df, similarity_matrix):
    """Finds the movie in the dataset and returns the top 10 matches."""
    all_titles = df['title'].tolist()
    
    # Handle typos using difflib
    matches = difflib.get_close_matches(movie_title, all_titles)
    
    if not matches:
        return None
    
    best_match = matches[0]
    movie_index = df[df.title == best_match].index[0]
    
    # Get similarity scores for this movie against all others
    scores = list(enumerate(similarity_matrix[movie_index]))
    
    # Sort by score (index 1) in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]
    
    recommendations = []
    for item in sorted_scores:
        recommendations.append(df.iloc[item[0]]['title'])
        
    return best_match, recommendations

def main():
    # File path - make sure movies.csv is in the same folder
    data_path = 'movies.csv'
    
    print("Initializing Recommender System...")
    movies_df = load_and_prep_data(data_path)
    sim_matrix = calculate_similarity(movies_df)
    
    user_input = input("\nEnter a movie you liked: ")
    result = get_recommendations(user_input, movies_df, sim_matrix)
    
    if result:
        match, recs = result
        print(f"\nSince you liked '{match}', you might also enjoy:")
        for i, title in enumerate(recs, 1):
            print(f"{i}. {title}")
    else:
        print("Sorry, we couldn't find that movie or anything similar.")

if __name__ == "__main__":
    main()