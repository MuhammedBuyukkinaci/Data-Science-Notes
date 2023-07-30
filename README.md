# Data-Science-Notes
Listing my Data Science Notes

1) We can access variables in pd.DataFrame().query() method like below

```
df = pd.DataFrame({'a':[1,2,3],'b':['x','y','z]})
BIGGER_FILTER = 2
df.query("a > @BIGGER_FILTER")
```

2) Usage of `.query()` can be encouraged. It is simple than complex filters.

3) For columns in the datetime format, use **parse_dates=['date_column_here']** in pd.read_csv().

4) Prefer dumping via to_parquet, to_feather, to_pickle instead of to_csv. It will preserver or data types and consume less spaces on hard disc.

5) We can use pd.DataFrame().style instead of MS Excel for formatting files.

6) validate option for pd.DataFrame().merge(validate=)

![merge_validate](./images/001.png)

7) Converting string columns which are categorical to category type is a best practice. We can do this via `.astype('category')`

8) [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) is a Python library that has useful visualizations for ML.

# Machine Learning Design Patterns

## The need for ML Design Patterns

1) ML models can be expressed in SQL using Google's BigQuery ML. It supports classification, regression and clustering.

2) Online prediction is useful if we care low latency. Batch prediction is useful if we don't care low latency.

3) Recommendation systems rely on batch predictions.

4) In Google's Big Query, Dataset is composed of many tables.

5) Data Quality can be checked in terms of accuracy, completeness, consistency and timel

## Data Representation Design Patterns

1) Input generally refers to non-processed data. When processed, it becomes feature.

2) Stepwise linear boundary function and Piecewise linear boundary function.

![](./images/002.png)

3) Optimizers generally work faster on [-1,1] range. Thus, it would be better to scale in that range. Linear Regression performed 9% faster when input is caled.

4) Scaling can be implemented after determining reasonable boundaries which exclude outliers.

5) When the distribution of the transformed input becomes normally distributed or uniform, it means transformation performed well.

6) Box-Cox Transformation is another way to deal with skewed data

```python

traindf['boxcox'], est_lambda = (
    scipy.stats.boxcox(traindf['num_views']))

# evaluation
evaldf['boxcox'] = scipy.stats.boxcox(evaldf['num_views'], est_lambda)

```

7) Modern ML algortihms don't require their inputs to be linearly independent. Modern ML algoritms use regularization such as L1 or L2 to prune redundant inputs.

8) Hashed Feature is a design pattern. Let's assume that we want to use departure airport in an ML model to predict whether the flight has a delay or not. However, in USA, there are 347 different airports. If we one-hot encode this, we will have 347 more columns, which consume a lot of memory and model spaces. Hence, it sounds reasonable to use a hash function to map these 347 values into some values. We can decide how many unique values in the transformed feature. Then, we can map input into features using a hash algorithm. This approach is durable to cold-start problem too. "A good rule of thumb is to choose the number of hash buckets such that each bucket gets about five entries".

9) Embedding is another design pattern. If we are creating an embedding layer, the size of embedding layer should be between fourth root of number of unique elements and 1.6\*square root of number of unique elements. Let's assume our vector has 625 unique elements. Our embedding vecor should be searched between 5(4th root of 625) and 40 (1.6*25).

10) [Feature columns](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html) is a detailed blog post by Tensorflow.

11) Feature cross means combining 2 or more categorical features into a single feature in order to capture the information between them. Complex models like Decision Trees and NN's can learn feature cross on their own. However, feature cross speeds up the learning. An example usage is to predict taxis fares by concatenating day of week(7) and hour of day(24). It will generate 168(7*24) new features. Hereby, we can capture some non-linearities. Always prefer concatenating non-correlated features.

12) We can use Feature cross and Embedding together.

13) Let's say we have a column named review whose possible values are [1,2,3,4,5]. 4 and 5 mean good reviews and 1,2,3 are bad review. Using raw form(1,2,3,4,5) and transformed form(good, bad) can contribute more to the model.

14) Embeddings and Bag of Words(BOW) are 2 different approaches to treat text. BOW doesn't take the order of the words into consideration.

## Problem Representation Design Patterns

### Reframing

1) Reframing means converting a regression problem to classificaiton or vice versa.

2) Quantile regression means predicting quantiles rather than real value. It is an implementation of reframing.

3) "A common rule of thumb for classification tasks is that we should have 10 times the number of model features for each label category. For a regression model, the rule of thumb is 50 times the number of model features".

4) Regression tasks require more data rather than classification tasks.

5) When to use regression and when to use classification

![](./images/003.png)

6) Hard Parameter sharing and Soft parameter sharing among NN's. Hard parameter sharing looks better compared to soft parameter sharing.

![](./images/004.png)

7) Multimodal distribution means the distribution has 2 peaks.

![](./images/005.png)

### Multilabel

8) Multilabel  can be xamplified in this way: One picture may have different animals and we want to predict whether the imge has a specific animal or not. In this design pattern, sigmod is used in the output layer rather than softmax. Softmax is used for multiclass classification.

9) The error of any ML model can be parsed into 3 parts: irreducible error, bias and variance.
Bias and variance are also called reducible error.

10) Stacking is an alternative to simply averaging different ML models. Stacking integrates an ML logic to combine predictions of base learners.

### Cascade

11) It is a design pattern meaning splitting the problem into sub-problems. Let's assume we want to predict if a transaction returns or not. However, resellers and retail buyers behave differently. Thus, we split the problem into the following steps:

- "Predicting whether a specific transaction is by a reseller

- Training one model on sales to retail buyers

- Training the second model on sales to resellers

- In production, combining the output of the three separate models to predict return likelihood for every item purchased and the probability that the transaction is by a reseller."

12) Another use case of Cascade Design Pattern is to use the output of first machine learning model as input of second machine learning model.

### Neutral Class

13) It is a design pattern that requires converting a binary classification problem into a 3-class multiclass classification. The new class is neutral class and it includes non-sure values.

14) It can be implemented in customer satisfaction models where a poll is having values between 1 and 10. Comments above 8 are regarded as good, comments below 5 are bad and the comments that lie between 5 and 8 are neutral.

### Rebalancing

1) "Machine learning models learn best when they are given a similar number of examples for each label class in a dataset".

2) Rebalancing means balancing classes. It can be carried in these ways:
    - Upsampling
    - Downsampling
    - Weighting

3) When we are evaluating our model, we shouldn't use upsampling. Let's assume we have a dataset whose 99 percent is 0 and 1 percent is 1. We upsampled the minority class to have a better training. We shouldn't evaulate our model on the upsampled data. We should have the original positive to negative ratio on evaluation data.

4) For an unbalanced data, it is more reasonable to use PR curve rather than AUC. For further details, take a look at [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)

5) When training a NN, assigning a custom value like `bias = log(num_minority_examples / num_majority_examples) make NN converge faster.

6) We can label anomally values in the following way:

    - Fit a clustering model on input data
    - Predict new data on the fitted model
    - Measure the distance between predicted result and cluster centers.
    - If distance is high, label it as anomally.

7) We can combine Downsampling and Class weights approach. We can also combine Downsampling with Ensemble can be carried out via building 9 different models on the data whose 90 percent is 0 and 10 percent is 0.
`



