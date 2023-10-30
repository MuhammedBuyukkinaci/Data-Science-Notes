# Practical Recommender Systems
The notes on this markdown file are taken from Practical Recommender Systems book.

# Chapter 1. What is a recommender?

1) There are 3 types of recommendations:

    - Non-personalized: All user viewing the same recommendation. A list of most popular movies on Netflix or most sold items on Amazon can be thoguht as non-personalized RS.
    - Semi-personalized: Based on geographical info, gender, nationality, whether user is moving or not. The RS doesn't know anything special to users. It is also called as segment-based RS.
    - Personalized: Amazon's for you is an example of personalized RS.

2) A watchlist/liked list on Netflix can be useful. It indicates interest.

3) Netflix recommendation algorithm takes time into consideration. The recommendations on mornings differ the recommendations on evenings.

4) Different terms on recommendations

![](./images/001.png)

5) How top picks work on Netflix

![](./images/002.png)

6) The rating prediction is only a part of RS(Recommendation Systems) but it is the most important one.

7) Dimensions to describe a RS

- Domain: "The domain is significant because it provides hints on what you’d do with the recommendations". The domain is determining whether RS should give recommendations more than once or not .

- Purpose: Measuring something different instead of our direct gal is considered as proxy goal. Netflix considers the amount of view as a deciding factor in order to evaluate its success, which is a proxy goal. The main goal is to increase revenues. However, using a proxy goal might lead to

- Context: It is the environment in which users get recommendations. Time of the day, location of users, the weather can be considered as context.

- Personalization Level

- Whose opinions

- Privacy and Trustworthiness

- Interfaces

- Algorithms

8) "Customers who bought this also bought this" is an example of seeded recommendations. These recommendations are based on a seed, which is the current item.

9) Organic presentation is that the website locates recommendations in an organic way like Netflix's homepage. Users just view movies chosen for them. Whereas, Amazon's Just For you is a non-organic way, which is a separate page listing items on Amazon.

10) Black box recommendations vs white box recommendations

![](./images/003.png)

11) For RS, generally there are 3 types of algorithms.

- Collaborative Filtering: Usage data like click, conversion and impression is used 

- Content Based Filtering: Content metadata and user profile used.

- Hybrid Recommenders: Mix of above two solutions.

12) MovieGeeks Architecture and main page layout

![](./images/004.png)

![](./images/005.png)

13) A baseline recommender system is to return popular items.

# Chapter 2: User behavior and how to collect it



