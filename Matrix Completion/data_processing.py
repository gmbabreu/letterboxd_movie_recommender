import pandas as pd
from tqdm import tqdm


def create_rating_matrix(df):
    # Get unique users and movies
    users = df['user_id'].unique()
    movies = df['movie_id'].unique()

    # Create an empty matrix with users as rows and movies as columns
    matrix = pd.DataFrame(index=users, columns=movies)

    # Iterate over the dataframe rows and populate the matrix
    for i, row in tqdm(df.iterrows()):
        user = row['user_id']
        movie = row['movie_id']
        rating = row['rating_val']
        matrix.loc[user, movie] = rating

    return matrix

def main():
    data = pd.read_csv("data/ratings_export.csv") 
    
    # Select necessary rows and drop null values
    data = data[["movie_id", "rating_val", "user_id"]].dropna()

    # Remove duplicated rows
    data = data[~data.duplicated(subset=["movie_id", "user_id"])]

    # Calculate the frequency of each movie
    movie_frequencies = data['movie_id'].value_counts()

    # Filter the dataframe to remove movies few occurences
    data = data[data['movie_id'].isin(movie_frequencies[movie_frequencies > 98].index)]

    # Calculate the frequency of each user
    user_frequencies =data['user_id'].value_counts()

    # Filter the dataframe to remove users few occurences
    data = data[data['user_id'].isin(user_frequencies[user_frequencies > 10].index)]

    # Create the rating matrix
    matrix = create_rating_matrix(data)

    matrix.to_csv('data/ratings_matrix.csv', index=False)

if __name__ == "__main__":
    main()