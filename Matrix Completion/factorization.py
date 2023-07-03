import pandas as pd
import numpy as np
from MatrixFactorization import MatrixFactorization



def main():
    ratings = pd.read_csv('data/ratings_matrix.csv')
   
    matrix = ratings.to_numpy()    
    matrix[np.isnan(matrix)] = 0

    
    # Create an instance of MatrixFactorization
    model = MatrixFactorization(matrix,  k=100, learning_rate=10**-5, l1=10**-3, l2=10**-3)

    # Train the model
    model.train(n_iterations=5000)

    # Get the factors U and V
    V = model.get_v().numpy().T

    # Get the full reconstructed matrix
    reconstructed_matrix = model.get_matrix()    

    # Create list of all movies
    movies = ratings.columns

    # Create a DataFrame from the matrix with movie names as column names
    pd.DataFrame(V, columns=movies).to_csv('data/movie_factor.csv', index=False)
    pd.DataFrame(reconstructed_matrix, index=ratings.index, columns=movies).to_csv('data/reconstructed.csv', index=False)



if __name__ == "__main__":
    main()
