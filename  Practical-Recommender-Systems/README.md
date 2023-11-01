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

1) 2 types of feedbacks

- Explicit: Ratings or likes

- Implicit: "Activity recorded by monitoring the user"

2) The purposes of RS might be different

- Amazon's purpose is to make people buy more

- GAP's purpose is to make you like and not cause you to return it

- Mofibo's(An online reader platform) purpose is let you open less books as much as possible.

- Netflix's purpose is to make you watch more films, especially their own products.

3) Event Lists on Netflix. An evidence collector is used to collect data like below.

![](./images/006.png)

4) A pageview can mean that the user is interested or it is lost in the website.

5) Page Duration on a page can mean different things like below.

![](./images/007.png)

6) Expansion linkes can refer to an interest.

![](./images/008.png)

7) Sharing on Social Media can be attributed to interest of an item.

![](./images/009.png)

8) Add to playlist, favorite list, watchlist indicates an interest.

9) Evidence collector(like xenn) should be a different app than our main app.

10) An example data model of event collector

![](./images/010.png)

11) The following data model is useful and flexible.

![](./images/011.png)

12) The following data model is a standard model.

![](./images/012.png)

# Chapter 3: Monitoring the system

1) We have to integrate an analytics app to measure the effect of RS.

2) We can't build a RS with an analytics dashboard to keep an eye on things.

3) Number of visitors, conversion rates, number of items sold are some important KPI's.

4) A conversion funnel of Amazon

![](./images/013.png)

5) An analytics dashboard example

![](./images/014.png)

6) "Having a dashboard that shows you how your site is doing will be a great help when doing recommender systems".

# Chapter 4: Ratings and how to calculate them

1) General pipeline of a RS

![](./images/015.png)

2) Having read the previous chapters, you should have considered the following:

- What’s the purpose of your site (the goals that you want users to achieve)?

- What events lead up to these goals?

- How many times has each of these events happened?

3) User-Item matrix can be thought as a dataset whose rows correspond to users and columns correspond to items.

![](./images/016.png)

4) An explicit matrix can be plotted on the matrix.

5) Amazon uses item-to-item recommendation algorithm because it has more than 200 million users. You can access to its details via [this link](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf).

6) Other websites can use browsing history of their users.

7) An example of "Recommended For You" of New York Times

![](./images/017.png)

8) Relevance should be taken into consideration in order to implement a RS.

9) Linkedin's "People You Know" and Facebook's "Friends Suggestion" are 2 RS based on implicit rating.

10) In e-commerce, rating happens after buying. This makes things complicated.

11) Time should be taken into consideration. More importance should be given to recent transactions/items.

12) Several events like conversion, click, viewing details etc. needs to be combined to constitute a user-item matrix.

13) In order to assign different weights to different events(details, moreDetails, genreView etc here), their impact on conversion should be taken into account.

`IRi,u = (w1 × #event1) + (w2 × #event2) + ... + (wn × #eventn)`












