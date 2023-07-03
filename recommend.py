import pandas as pd
import numpy as np
from user_ratings import get_ratings



def main():
    # Import V component
    V50 = pd.read_csv(r"C:\Users\gmbab\OneDrive\Área de Trabalho\letterboxd\movies_factor50.csv")
    V100 = pd.read_csv(r"C:\Users\gmbab\OneDrive\Área de Trabalho\letterboxd\movies_factor100.csv").to_numpy()



    # Create list of all movies
    movies = V50.columns
    V50 = V50.to_numpy()


    # Get the user's ratings (recommended user rating 30)
    user_rating = pd.read_csv(r"C:\Users\gmbab\OneDrive\Área de Trabalho\letterboxd\user1.csv").to_numpy().T[1].astype("int")
    #ratings = get_ratings(input)

    # Find prediction by reconstructing the row

    from  tqdm import tqdm
    P = [[10**-3, 10**-3]]
    for p in P:
        print("50", p)
        E = []
        for i in tqdm(range(40)):
            prediction, er = reconstruct(user_rating, V50, l1=p[0], l2=p[1])
            E.append(er)
        print(np.mean(E), np.median(E), min(E), max(E), np.std(E))
    
    for p in P:
        print("100", p)
        E = []
        for i in tqdm(range(40)):
            prediction, er = reconstruct(user_rating, V100, l1=p[0], l2=p[1])
            E.append(er)
        print(np.mean(E), np.median(E), min(E), max(E), np.std(E))
        print(E)
    
"""
    # Find the indices that a have not been rated (zero indices)
    predictions = []
    errors = []
    for i in range(50):
        p, e = reconstruct(user_rating, V, l1=10**-3, l2=10**-3)
        predictions.append(p)
        errors.append(e)
    prediction = predictions[np.argmin(errors)]
    
    prediction, e = reconstruct(user_rating, V)
    prediction[np.nonzero(user_rating)] = 0

    # Sort the predicted ratings
    df = pd.DataFrame({'Movie': movies, 'Rating': prediction})
    df = df[df['Rating'] != 0].sort_values(by='Rating', ascending=False)
    print(df)
    #print(min(errors))
    #print(np.mean(errors), np.std(errors))
    """
    


def reconstruct(r, V, learning_rate=10**-7, n_iterations=15000, l1=0, l2=0):
    k = V.shape[0]
    
    U = np.random.rand(1, k)/100  # Initialize U with random values

    non_zero_indices = np.nonzero(r)  # Find the indices where r is not zero
    
    error = [10**4]
    old_error = [10**4]

    for n in range(n_iterations):
        # Break if error does not decrease
        if sum(old_error)<sum(error) and n>100: 
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

    return prediction, sum(error)

if __name__ == "__main__":
    main()