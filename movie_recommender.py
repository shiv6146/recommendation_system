import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings; warnings.simplefilter('ignore')

# Lambda used to normalize a given movie title
def normalize_title(title):
    return ''.join(c for c in title if c.isalnum()).lower()

# Read movies dataset
movies = pd.read_csv('links_small.csv')

# Read movies metadata csv
movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Create a sparse matrix with movie id and tmdbId
movies = movies[movies['tmdbId'].notnull()]['tmdbId'].astype('int')

# Filter out rows with inappropriate characters
movies_metadata = movies_metadata.drop([19730, 29503, 35587])

# Convert id column to int
movies_metadata['id'] = movies_metadata['id'].astype('int')

# Create a new data frame which is a subset of movies in movies_metadata.csv available in links_small.csv
md_movies = movies_metadata[movies_metadata['id'].isin(movies)]

# Filter and combine tagline and overview columns together into a new description column
md_movies['tagline'] = md_movies['tagline'].fillna('')
md_movies['overview'] = md_movies['overview'].fillna('')
md_movies['description'] = md_movies['overview'] + md_movies['tagline']
md_movies['description'] = md_movies['description'].fillna('')

# We compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document.
# This will give you a matrix where each column represents a word in the overview vocabulary 
# (all the words that appear in at least one document) and each column represents a movie
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

# TF-IDF score is the frequency of a word occurring in a document, down-weighted by the 
# number of documents in which it occurs
tfidf_mat = tfidf.fit_transform(md_movies['description'])

# Calculating the similarities can be done with cosine similarities
# As we have tfidf_matrix already at hand we can use linear_kernel to directly compute dot product
similarities = linear_kernel(tfidf_mat, tfidf_mat)

# Add index column to provide continous indexing to new data frame (md_movies)
md_movies = md_movies.reset_index()

# Reverse index movie titles to movie id
titles = md_movies['title'].copy()
md_movies['title'] = md_movies['title'].apply(lambda x: normalize_title(x))
indices = pd.Series(md_movies.index, index=md_movies['title'])

# Takes in a movie title and returns a list of similar movies based on the movie description
def get_recommendations(title):
    title = normalize_title(title)
    try:
        idx = indices[title]
        scores = list(enumerate(similarities[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # Getting top 10 movies with high similarity scores other than passed movie title itself
        scores = scores[1:11]
        top_indices = [x[0] for x in scores]
        return titles.iloc[top_indices]
    except Exception:
        print "Oops! I have not heard of that movie yet!"
        return

recommended_movies = get_recommendations(raw_input("Enter a movie name: "))
if recommended_movies is not None:
    print "Recommended movies with similar plot:"
    for mov in recommended_movies:
        print mov
