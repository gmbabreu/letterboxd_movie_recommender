# Letterboxd Movie Recommender
This program recommends movies based on the user's movies tastes. It is made to work with the Letterboxd website, you can input your Letterboxd username and it will provide recommendations. If you do not have a letterboxd account, you can also manually fill in the user.csv file with movie ratings.
The recommender uses publicly available movie rating data from letterboxd. It performs matrix factorization to condense those ratings and their features into the movies_factor.csv file, from which it bases its recommendations when a user inputs his own movie ratings


## Setup
The recommender only requires that you download 3 files for it to work: recommend.py, movies_factor.csv, and either get_user.py or user.csv. The get_user.py scrapes your movie ratings from letterboxd, while user.csv file can just serve as a substitute by having the ratings manually written in. All the files inside the Matrix Completion folder are not necessary if all you want is the recommendations, but you can use them to alter the Matrix Factorization process. 
If you intend on scraping your ratings from letterboxd, all you have to do is download recommend.py, movies_factor.csv, and get_user.py. Go into recommend.py and write your letterboxd username in USERNAME.
If you do not have a letterboxd account, download recommend.py, movies_factor.csv, and user.csv. Go into the user.csv file and manually write down movie ratings based on movies in the file.

Finally you need to have both pandas and numpy libraries installed. If you downloaded the get_user.py, you must also have BeautifulSoup and Requests library installed.

Once all of this is done, you can go into the recommend.py file and run it



