# Practical Recommender Systems

The notes on this markdown file are taken from [Practical Recommender Systems](https://www.goodreads.com/book/show/28510003-practical-recommender-systems?ac=1&from_search=true&qid=3Wyf6zSZ92&rank=1) book.

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

![](./images/019.png)

12) Several events like conversion, click, viewing details etc. needs to be combined to constitute a user-item matrix.

13) In order to assign different weights to different events(details, moreDetails, genreView etc here), their impact on conversion should be taken into account.

`IRi,u = (w1 × #event1) + (w2 × #event2) + ... + (wn × #eventn)`

![](./images/018.png)

14) A music streaming platform like Spotify might give more importance to recent activities.

15) "The thinking is that if a user buys an item that’s popular, it doesn’t provide much information about the user’s taste. If the same user likes something only few people like, then it could be a better indication of the personal taste of the consumer".

16) Assigning more weights to less frequent items could boost our RS.

![](./images/020.png)

# Chapter 5: Non-personalized recommendations

1) Coupon.com's non-personalized RS on mainpage. It recommends

2) A simple RS is to order what people are most likely to favor. This might be price-oriented, recency-oriented. 

3) Top 10 Food Restaurants on a meal website is a non-personalized RS. It was used in the past a lot.

4) A RS has 2 components: builder(training) and serving(inference).

5) A RS can be either memory-based or model based. Memory based RS can be challenging.

6) Some websites have a fallback recommender in case of safety mechanism besides their main RS.

7) It is a good practice to separate RS from the main app.

8) Frequently Bought Together(FBT) is an example of seeded recommendations. "People who viewed this item also viewed these" is also seeded recommendation.

9) FBT might not work well frequently. In addition to a rare item in a cart, there might be a popular item in a cart. Thus, popular item will be promoted irrespective of the content of the cart. This is challenging for FBT. Most people are buying milk from the supermarket. FBT will result in a boost in the recommendation of Milk.

![](./images/021.png)

10) Association rules in FBT

![](./images/022.png)

11) For a transaction like below, In order to find association rules, we should find frequency sets. Association rules work with explicit ratings(buys etc) better than implicit ratings(clicks).

![](./images/023.png)

- Confidence: 

![](./images/024.png)

![](./images/025.png)

- Support:

![](./images/026.png)

![](./images/027.png)


12) A new type of recommendation: "What other items do customers buy after viewing this item?"

![](./images/028.png)

13) "Online Consumer Behavior"  by Angeline G. is a book focusing on how users behave on the internet.

# Chapter 6: The user (and content) who came in from the cold

1) Cold start problem exists not only for new users but also for new items.

2) Some common connections if a user liked movie #1 in order to recommend movie #2

![](./images/029.png)

3) A solution to cold start problem by Netflix for new items.

![](./images/030.png)

4) Scientific papers state that personalized recommendations can't be calculated before a user rates 20 to 50 items.

5) Amazon's "What other people are looking" is a recommendation method.

6) Gray sheep can be considered as a unique user in terms of RS.

7) A solution to cold-start problem is direct users to log in with Google and Facebook accounts.

8) Sometimes, it is better to implement a business logic after recommendations. For example, RS in online movie market output a list of recommendation. The focus movie is a horror film and the recommendation is partially composing of sci-fi movies. It is better to drop sci-fi movies from the list.

![](./images/031.png)

9) Demographic Recommendations is to recommend content popular with that segment. It is an implementation of  semi-personalized RS.

10) Some obvious segments 

- Age

- Gender

- Location

11) Some not-so-obvious segments

- German women who browse in the evening during the weekend and like action

- American male teenagers who buy horror films during school time

12) A handy method to deal with cold start and sparsity in data is to abstract item into its category.

![](./images/032.png)

13) Improve your recommendations section by Amazon. It is implemented for logged-in users, not for anonymous users.

![](./images/033.png)


14) "Active learning for recommender systems is about creating an algorithm that comes up with good examples for the user to rate, which then provides the recommender with valuable information about the person’s preferences.".

15) Association rules can be used in personalized recommendations too. It is an easy and quick way to add personalization.

16) Viewing a single item is sufficient to show "recommended for you" RS.

# Chapter 7: Finding similarities among users and among content

1) Clustering can halep in collaborative filtering.

2) Netflix's More Like This rs

![](./images/034.png)

3) Similarity can be calculated among content and among users using different metrics.

4) In the literature, few similarity functions are outputing good results.

5) Similarity functions affect similarity scores

![](./images/035.png)

6) Different types of data: Unary data, Binary data, Quantitative data

![](./images/036.png)

7) Jaccard Similarity between 2 items on a user-item matrix

![](./images/037.png)

8) A simple Similarity score for one movie rating between 2 users. 

![](./images/038.png)

9) This can be called as L1-norm, manhattan distance or taxicab geometry. We just take absolute difference between 2 data points.

![](./images/038.png)

10) L2-norm known as Euclidean norm

![](./images/039.png)

11) Cosine similarity between 3 persons among 2 movies.

![](./images/040.png)

12) Amazon uses item to item collaborative filtering because the number of users is much more than the number of items on Amazon.

13) Vanilla cosine similarity

![](./images/041.png)

14) Adjusted cosine similarity, which is normalized by average rating of user. It ranges from -1 to +1.

![](./images/042.png)

15) Pearson correlation, which ranges from -1 to +1.

![](./images/043.png)

16) Pearson correlation and adjusted cosine similarity are so close terms to each other. Adjusted cosine similarity takes items rated by either or both users, whereas pearson correlation takes items rated by only both users.

17) We have to use clustering in order to compute user-user similarity efficiently.

18) It is difficult to make Kmeans Clustering work correctly.

19) User collaborative filtering Venn Diagram.

![](./images/044.png)

20) [Grouplens](https://grouplens.org/) is a research group in University of Minnesota.

21) It is a good practice to filter 20-50 users in order to compute similarity metric(pearson correlation or cosine similarity or adjusted cosine similarity) rather than whole data.

22) It is a good way to double check clustering with pearson correlation. The points in the same cluster should have a high pearson correlation pairly.

# Chapter 8: Collaborative filtering in the neighborhood

1) Collaborative Filtering(CF) is an umbrealla of methods. It also covers neighbor-based filtering. CF uses only ratings(explicit or implicity) as recommendation engines, which means no metadata is used. CF is content-agnostic.

2) Amazon's "Recommended For You" is based on CF.

3) Neighbor based filtering can be handled in 2 ways:

- User based Filtering

- Item based Filtering

![](./images/045.png)

4) User based CF pipeline

![](./images/046.png)

5) Item based CF Pipeline

![](./images/047.png)

6) If you want to calculate as few similarities as possible and you have much more users than items, you should prefer item-based filtering.

7) In terms of serendipity, user-based filtering sounds more reasonable.

8) In terms of explainibility, item-based filtering sounds more reasonable because it is easy to state that you may like this because you bought that etc. Whereas, user-based filtering is hard to explain while preserving privacy.

9) "A way to implement collaborative filtering is by first finding all the items that are rated by several users (more than two users at least)"

10) User who have rated only 1 or 2 items shouldn't be taken into consideration for CF. Users who rated lots of items may lead to too generalized recommendations.

11) Clustering might be used to narrow down the search space.

12) How to approach the similarity scores: Threshold or Top-N. Both have pros and cons.

![](./images/048.png)

13) Classification way and regression way for rating

![](./images/049.png)

14) Some of candidate items are recommended, this is based on finding item similarities and then computing ratings for each candidate.

![](./images/050.png)

15) How to compute the rating of a possible item for a user. Normalized user ratings must be used.

![](./images/051.png)
![](./images/052.png)

16) In order to get the prediction rating of an item for a user, all possible candidates must be rated thanks to similarity scores. After calculating ratings, return top 10.

17) Some ways to overcome cold start problems:

- Let users rate a few items

- Create a new arrivals list

18) Another way to deal with cold start problem is exploit/explore strategy.

![](./images/053.png)

19) Online and offline serving

![](./images/054.png)

20) [`coo_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html) is a way to build a user-item matrix.

![](./images/055.png)

21) How rating is calculated after calculating similarities.

![](./images/056.png)

22) Some considerations while implementing CF

![](./images/057.png)

23) Some drawbacks of CF

- Sparsity

- Gray Sheep

- Number of Ratings: We need many ratings to have a good CF model.

- Similarity: CF follows users' behavioral trends. CF tends to put most popular items into recommendations often.






























