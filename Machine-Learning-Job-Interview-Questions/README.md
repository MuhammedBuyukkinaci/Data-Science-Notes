# Machine-Learning-Job-Interview-Questions
Containing questions that can be asked in a ML interview.

1) A company says they are scoring test data in the database not in RAM, which means no pickle file is used to score new data. Which ML algorithm is most likely used by that company?

- Linear Regression

- Logistic regression

2) Let's say we have 2 datasets. Both have binary targets. Target ratio of first dataset is 1:99. Target ratio of second dataset is 10:90. For both, we have 0.80 AUC. On which dataset we are more successfull?

- We are more successfull on 10:90 dataset

3) Which ML algorithm may contain more than one target column during training? Describe briefly.

- Neural Networks

4) What is the difference between Random Forest and Gradient Boosted Trees?

- The former is running decision trees in a parallel manner.
- The latter is running decision trees based on previous ones.

5) Which machine learning algorithm works best with categorical features which have high cardinality?

- Gradient Boosting trees

6) How do you feed categorical variables into Neural Networks for tabular data?

- Embedding layer

- One hot encoding

7) How do you handle null values if you want to train a NN?

- Median/Mean Filling

- Null flagging, which means creating a separate column saying whether the values is null or not.

8) Which ML algorithm is more preferrable when your dataset is relatively small and the number of features are high?

- SVM sounds logical. Linear and logistic regressions are also reasonable.

9) How could you understand whether there is almost no signal in a classification problem?

- Replace input data with noisy data(np.random.randn) and feed it. If losses are similar, there looks almost no signal.

- Look at distribution of predicted values. If its standard deviation is small, there appears so little signal.

10) What kind of problems have little signal by their nature?

- Stock price prediction has so little or 0 signal. It is based on Random Walk Hypothesis.

11) You are asked to train 6 models on different datasets and targets as follows

    | Training Years            | Target   | Model_name  |
    | --------------------------|:--------:| -----------:|
    | 2011-2012-2013-2014-2015  | 2016     | model01.pkl |
    | 2012-2013-2014-2015-2016  | 2017     | model02.pkl |
    | 2013-2014-2015-2016-2017  | 2018     | model03.pkl |
    | 2014-2015-2016-2017-2018  | 2019     | model04.pkl |
    | 2015-2016-2017-2018-2019  | 2020     | model05.pkl |
    | 2016-2017-2018-2019-2020  | 2021     | model06.pkl |

Then, you are requested to choose 2 different models to blend your predictions. Which model pairs do you choose?

- model01 and model06 because data that these models are trained on are most different for all possible pairs.