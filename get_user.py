import pandas as pd
import requests
from bs4 import BeautifulSoup


def scrape_ratings(username):
    # Construct the URL for the user's ratings page
    url = f"https://letterboxd.com/{username}/films/ratings/"
    response = requests.get(url)
    
    ratings = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find the total number of pages
        pagination = soup.find("div", class_="paginate-pages")
        if pagination:
            total_pages = int(pagination.find_all("a")[-1].text)  
        else: 
            total_pages = 1
        
        # Loop through each page and scrape the ratings
        for page in range(1, total_pages+1):
            page_url = f"https://letterboxd.com/{username}/films/ratings/page/{page}/"
            page_response = requests.get(page_url)
            if page_response.status_code == 200:
                page_soup = BeautifulSoup(page_response.content, "html.parser")
                rating_elements = page_soup.find_all("li", class_="poster-container")
                for rating in rating_elements:
                    # Extract the film title from the data attribute of the <div> element
                    film_title = rating.find("div")["data-film-slug"].split("/")[2]
                    
                    # Check if the film has a rating
                    rating_element = rating.find("span", class_="rating")
                    if rating_element:
                        # Extract the rating value from the last character of the last class name
                        film_rating = int(rating_element["class"][-1][-1])

                        # if a zero was extracted it is supposed to be a ten
                        if film_rating == 0: film_rating=10
                        
                        # Append the film title and rating as a tuple to the ratings list
                        ratings.append((film_title, film_rating))
            else:
                print(f"Error occurred while scraping page {page}. Skipping...")
    else:
        print("Invalid username or error occurred.")
    
    # Create a DataFrame from the ratings list
    df = pd.DataFrame(ratings, columns=["movies", "user ratings"])
    
    return df



if __name__ == "__main__":
    username = "gmbabreu"
    ratings = scrape_ratings(username)
    ratings.to_csv("user.csv", index=False)