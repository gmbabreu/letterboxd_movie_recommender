import pandas as pd
import numpy as np
from MatrixFactorization import MatrixFactorization



def main():
    ratings = pd.read_csv(r"C:\Users\gmbab\OneDrive\Área de Trabalho\letterboxd\Matrix Completion\data\ratings_matrix.csv")
   
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
    pd.DataFrame(V, columns=movies).to_csv(r'C:\Users\gmbab\OneDrive\Área de Trabalho\letterboxd\movies_factor.csv', index=False)
    pd.DataFrame(reconstructed_matrix, index=ratings.index, columns=movies).to_csv(r'C:\Users\gmbab\OneDrive\Área de Trabalho\letterboxd\Matrix Completion\data\reconstructed.csv', index=False)



if __name__ == "__main__":
    main()

"""
K = 10
Final
15312404.0,     6660
15297016.0,     10000 


K = 20
14591841.0,             2500
14159815.0,             5000
14006112.0,             10000,  7:28

k = 50
12995527.0,             2600
12??????,               5000


K = 100
13950364.0,             1000
11825616.0,             2000
9952769.0,              5000












learning_rate=10**-5, l1=0, l2=10**-3

learning_rate=10**-5, l1=0, l2=10**-2

learning_rate=10**-5, l2=0, l1=10**-2

  """