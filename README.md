# PROJECT PHASE 4

### Overview
Our goal in this project is to deliver personalized suggestions or recommendations to users given the dataset provided to us by The University of Stanford. The dataset exists in this folder.

![Repository Image](images/recommended.jpg)


## 1. Business Understanding 

Often a times, we have found ourselves searching and parsing on streaming sites to find movies to watch which are similar to one we have just watched. This can be time-consuming and always frustrating and sometimes even giving up completely to even find anything worth to watch.

This is when recommendation systems come in hand as they reduce the time-consuming, manually searching for content to watch and use advanced machine learning algorithms to recommend contents based on user data.

Recently a streaming company by the name JogooFlix approached us asking for a working recommendation system model and also highlighted the problems their customers go through without the recommendations systems.

Since they didn't want to violate the terms and agreements of their user-policy, we used the popular MovieLens dataset available online to train and  model our recommendation systems.

We will be coming up with several models using algorithms such as Collaborative and Content-based which come in several popular libraries such as pyspark, scikit and surprise.Without further ado let's try to delve into the understand our data.


## 2. Data Understanding

This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.

The data are contained in the files `links.csv`, `movies.csv`, `ratings.csv` and `tags.csv`. But for the sake of this project and we only used to csv files namely
`movies.csv` and `ratings.csv`

In order to understand our data, we will try to see its shape , datatypes , information it contains by simply calling the `.head()` method and maybe some visualizations.

The columns of the movies dataset has both string and  numeric data types, it appears to not having missing any data points with 9742 entries and 3 columns namely : `movieId`, `title` and  `genres`. 

###### Movies.csv 
The most occuring genre is drama as observed in the graph below while there are **951** unique combinations of genres the movie `Eros (2004)` appears to have been recorded twice in the dataset.

|       |title |genres
|count	|9742|9742
|unique	|9737|951
|top	|Emma (1996)   |Drama
|freq	|2      |1053


<p align="center">Graph of Counts of Genres </p>
<p align="center">
  <img src="C:\Users\hp\phase-4-project\images\genres.png" alt="genres">
</p>

###### Ratings.csv

The Ratings Dataset has 100836 entries with 3 columns namely `userId`, `movieId`, `rating ` and `timeamp`t. All of the columns are in numeric datatypes with no null data entries.

The average rating is about 3.5 with a standard devistion of 1 with the minimum being 0.5 and maximum being 5.The numbers of users who rated are 610 in total  0

<p align="center">Graph of Ratings Distribution </p>
<p align="center">
  <img src="C:\Users\hp\phase-4-project\images\ratings.png" alt="genres">
</p>

### Data Preparation 

There was not much to do here as our data was in desires form

### I. Collaborative Filtering
Collaborative filtering is the process of filtering for information or patterns using techniques involving collaboration among multiple agents, viewpoints, data sources. Basically, it is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users.

There are 2 approaches to Collaborative Filtering:

1) Memory-Based Collaborative Filtering - It is an approach which finds similarity between users or between items to recommend similar items. Examples include Neighbourhood-based CF and Item-based/User-based top-N recommendations.

2) Model-Based Collaborative Filtering(Surprise-library) - In this approach we use different data mining, machine learning algorithms to predict users' rating of unrated items. Examples include Singular Value Decomposition (SVD).

3) Alternating Least Squares (ALS-using Pyspark)

### Evaluations 
We evaluated the model-based CF of various models by calculating their rmse values.
Here's the following table:

### Evaluation Table



|Model Based Collaborative Filtering Evaluation|              |
|----------------------------------------------|--------------|
|**Models**                                    |**RMSE Values**|
|KNN Basic (Surprise-library)                  |0.978         |
|KNN WithMeans(Surprise-library)               |0.907         |
|KNN Baseline (Surprise-library)               |0.883         |
|SVD (Surprise-library)                        |0.875        |
|ALS(Pyspark)                                  |0.875        |


Here is an example code and how pyspark was able to generate recommendations for a certain user:
```
def recommend_for_specific_user(user , df , users_df):
    u_df = users_df.toPandas()
    rate = df.toPandas() 

    movie_ids = []
    movies_and_rates = []
    for i in u_df[u_df['userId'] == user]['recommendations']:
        for x in i:
            movie_ids.append(x[0])
            movies_and_rates.append(x[1])
            
    titles = []
    for x in movie_ids:
        if x in rate['movieId']:
            titles.append(rate[rate['movieId'] == x]['title'].drop_duplicates().values[0])
    return  list(zip(titles,movies_and_rates))  
```
and recommended movies were:
```
("Adam's Rib (1949)", 5.128778457641602),
 ('Beautiful Thing (1996)', 4.945532321929932),
 ('Chorus Line, A (1985)', 4.893959999084473),
 ('Crossing Delancey (1988)', 4.880897045135498),
 ('Lady Jane (1986)', 4.880897045135498),
 ('Guys and Dolls (1955)', 4.853848934173584),
 ('Wallace & Gromit: The Best of Aardman Animation (1996)', 4.832089900970459),
 ('Six Degrees of Separation (1993)', 4.82675313949585),
 ('Love and Death (1975)', 4.824601650238037),
 ('Saving Face (2004)', 4.81075382232666)
```