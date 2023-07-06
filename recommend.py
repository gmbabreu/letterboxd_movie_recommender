import pandas as pd
import numpy as np

SCRAPE = True
USERNAME = "gmbabreu"


def main():
    # Import V component
    V = pd.read_csv(r"C:\Users\gmbab\OneDrive\Área de Trabalho\letterboxd\movies_factor.csv")

    # Create list of all movies
    movies = V.columns
    V = V.to_numpy()

    # Get the user's ratings (recommended minimum amount of ratings: 30)
    if SCRAPE:
        from get_user import scrape_ratings
        user = scrape_ratings(USERNAME)
    else:
        user = pd.read_csv("user.csv")

    # Put the movies in order by merging movies DataFrame with user ratings DataFrame
    movies_df = pd.DataFrame({'movies': movies})
    merged = pd.merge(movies_df, user, on='movies', how='left').fillna(0)
    ratings = merged["user ratings"].to_numpy()

    # Recontrust the ratings row
    prediction = reconstruct(ratings, V, l1=10**-3, l2=10**-3)
    
    # Sort out movies that have been rated
    prediction[np.nonzero(ratings)] = 0

    # Sort the predicted ratings
    df = pd.DataFrame({'Movie': movies, 'Rating': prediction})
    df = df[df['Rating'] != 0].sort_values(by='Rating', ascending=False)
    print(df.head(20))
    
def reconstruct(r, V, learning_rate=10**-6, n_iterations=50000, init=10, l1=0, l2=0):
    k = V.shape[0]

    U = np.random.rand(1, k)/init  # Initialize U with random values

    non_zero_indices = np.nonzero(r)  # Find the indices where r is not zero
    
    error = [10**4]     # Define error to be initially really high
    old_error = [10**4]

    for n in range(n_iterations):
        # Break if error does not decrease
        if sum(old_error)<sum(error) and n>10: 
            break
        
        # Save new error
        old_error = error

        # Compute the predicted value
        prediction = U.dot(V).flatten()
            
        # Compute the squared error based on non-zero values in r
        error = (r[non_zero_indices] - prediction[non_zero_indices])**2

        for j in range(k):
            grad_j = 2*np.dot(V[j,non_zero_indices],error) - l2/2 * U[:, j]
            U[:, j] += learning_rate * grad_j - learning_rate * l1 * np.sign(U[:, j])
            
    return prediction

if __name__ == "__main__":
    main()
