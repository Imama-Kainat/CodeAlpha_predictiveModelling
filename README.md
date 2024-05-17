# CodeAlpha_predictiveModelling

Developed a Linear Regression model to predict housing prices in Boston based on various features. Achieved a moderate model fit with an R-squared value of 0.67, demonstrating strong skills in data analysis, machine learning, and predictive modeling.


Its complete analysis of the Boston Housing dataset using linear regression. It uses the scikit-learn library to load the dataset, split it into training and test sets, train a linear regression model, make predictions, and evaluate the model's performance.


1. Import the necessary libraries: The script begins by importing the required libraries, including `fetch_openml` from `sklearn.datasets` to load the Boston Housing dataset, `train_test_split` from `sklearn.model_selection` to split the data into training and test sets, `LinearRegression` from `sklearn.linear_model` to create and train a linear regression model, `mean_squared_error` and `r2_score` from `sklearn.metrics` to evaluate the performance of the model, and `pandas` to handle the data as a DataFrame.

2. Load the dataset: The script loads the Boston Housing dataset using the `fetch_openml` function and converts it to a pandas DataFrame.

3. Split the data: The script splits the dataset into features (X) and the target variable (y) and further splits the data into training and test sets using the `train_test_split` function.

4. Initialize the model: The script initializes a linear regression model using the `LinearRegression` class.

5. Train the model: The script trains the model on the training set using the `fit` method.

6. Make predictions: The script makes predictions on the test set using the `predict` method.

7. Evaluate the model: The script calculates the mean squared error (MSE) and R-squared value using the `mean_squared_error` and `r2_score` functions, respectively.

8. Display the results: The script prints the MSE and R-squared value to evaluate the performance of the model.

Overall, the code demonstrates a complete workflow for performing linear regression analysis on the Boston Housing dataset, from loading the data to evaluating the model's performance.
