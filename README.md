# Machine Learning: Decision Trees and Random Forests

In this repository, I explore machine learning libraries and techniques, particularly focusing on Decision Trees and Random Forests.

## Lab Description

For this lab, I'll be working with the heart.csv dataset. The primary objective is to understand Random Forest Classification by following a tutorial provided in a video linked below. It's recommended to watch the video between the 5:00 minute and 19:00 minute marks to grasp all the steps involved, including manual hyper-parameter tuning.

[Watch Video Tutorial on Random Forest Classification](https://www.youtube.com/watch?v=BXkqEXjBf5s&ab_channel=MachineLearningLinks)

During the lab, I'll vary the number of default estimators for the classifier, known as `n_estimators`. This parameter determines the number of random trees used by the ensemble classifier. My task is to write code to find the number that offers the best accuracy on the test data (`X_test`). I'll report the scores obtained by `clf.score(X_test, Y_test)` over a wide range of `n_estimator` values.

## Instructions

1. Download the dataset `heart.csv`.
2. Follow the tutorial video provided in the link.
3. Implement manual hyper-parameter tuning as demonstrated in the video.
4. Write code to search for the optimal number of estimators that maximize accuracy on the test data.
5. Summarize the results in a concise data table.

## Example Output

Your output should include a data table summarizing the accuracy scores obtained for various `n_estimator` values.

