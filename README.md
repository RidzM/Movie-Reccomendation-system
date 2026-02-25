
# Movie Recommendation System 🎬

## Project Overview

This project was developed during my internship to solve the problem of "information overload" when choosing a movie. I built a **Content-Based Recommendation Engine** that suggests movies by calculating metadata similarity. Unlike simple popularity-based lists, this system looks at the "DNA" of a movie—its genre, cast, and director—to find specific matches for a user's taste.

## The Dataset

The project uses a dataset consisting of roughly **4,800 movies**. To ensure high-quality recommendations, I performed feature selection to focus on the most descriptive attributes:

* **Primary Metadata**: Genres, Keywords, and Taglines.
* **Personnel**: Lead Cast and Directors.

## Key Features & Logic

* **Fuzzy Matching**: I integrated the `difflib` library so the system can handle typos (e.g., if a user types "bat man", the system correctly identifies "Batman").
* **NLP Pipeline**: I used `TfidfVectorizer` to convert raw text into a mathematical matrix, removing common "stop words" that don't add value to the recommendation.
* **Similarity Scoring**: The engine uses **Cosine Similarity** to measure the angle between movie vectors. The closer the vectors, the higher the recommendation rank.

## Tech Stack

* **Language**: Python 3.x
* **Data Handling**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (TF-IDF, Cosine Similarity)
* **Text Processing**: Difflib

## How to Run the Project

1. **Clone the repository** and ensure `movies.csv` is in the root directory.
2. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn

3. **Execute the recommender**:
bash
python movie_recommender.py



## Author
Riddhi Mehta





