# **Stats Companion** _(WORK IN PROGRESS)_

My program tries to support all those people who are not familiar with the methods of statistical analysises but they can utilize them during their job or any other parts of their life. 

## **Guide**
This program was built to work automatically, to start you have to pass in the name(s) of the database(s). The train database should be the second command-line argument (after project.py), which helps the program to learn patterns to make predictions on the test database. Test database has to be the third command-line argument. All in all your prompt should look something like this:

`python project.py train_dataset.csv test_dataset.xlsx`

### ***Given databeses must only consist of the variable we want to make a prediction for and those variables that used to make the prediction!***
For example if we want to predict a student's score on a test:
Train database should contains the score of the test and other independent variables while test database should contain only the independent variables **without the variable you want to predict**.

Independent Variable: This is the variable used by a model to predict or explain the value of the outcome variable, and in machine learning, these are known as features or attributes.
Dependent Variable: This is the variable whose value is intended to be predicted, often called the target variable in machine learning, as it represents the outcome resulting from changes in the independent variables.

Although it is not necessary to give a test dataset, it can be beneficial when you have to estimate the price of a house according to a house database patterns. **It is important to have the same columns in the test dataset as in the train** otherwise the program won't work.

After the script was run the program will ask automatically whether you would like to get details of your dataset. Finally you have to choose the statistical method you would like to implement on your dataset, select the target variable and the program will output the results of the analysis.

## **Functions**
### Data_reader
The `data_reader()` function ensures that the file name(s) which were given in the command-line arguments going to be transformed into pandas dataframes. This function can handle CSV and Excel files with the help of `pd.read_csv` and `pd.read_excel`.

### Cleaning
The cleaning() function has the most functionality all of my functions. 
`cleaning(tanito, X_test = 0, y_variable = 0, make_dummies = False, y_type = "continuous", X_Var = 0, y_Var = 0)`

This function can handle a maximum of 4 datasets `tanito` is the full train database, `X` contains all the input variables while `y` has only that variable we would like to make a prediction for, finally X_test is the predictive dataset.
Except `tanito` all of them has a default _0_ as during descriptive statistical analysis the program doesn't use them.

With the help of `make_dummies` argument you can choose if you want to make dummies from categorical columns or not. This feature was necessary since this function is also used in the statistical analysis as well.

The function was constructed to be able to handle three types of dependent variables through the `y_type` argument that's default value is "continuous".
**When to use this:** if want to estimate a numberlike variable, _such as the price of a car according to it's features_. 
On the other hand if the variable we would like to predict is binomial or multinomial (*so it has yes/no or categorical values*) it is advised to change the `y_type`'s value.
**When to use "binary" in `y_type`:** in the case of a binomial variable, _for instance when want to estimate whether company will go bankrupt or not_.
**When to use _"categorical"_ in `y_type`:** when the dependent variable has more than two categories, _such as if we want to predict what grade a student will get on his/her test_.

### Descriptive_stat
Responsible for making and presenting the descriptive statistics. It outputs a written analysis about the measures, in addition also saves all the statistical measures in a csv file so users can insert it into another report if it is needed.

### Prediction
Prediction function provides the opportunity to estimate the value of the target variable according to the input variables. Thanks to `cleaning()` function it is able to predict three kinds of variable:

- Continuous variable by using linear regression on all the variables that were given in the input database. The multicollinearity among the independent variables is handled by price component analysis.

- Binary variable by using logistic regression with all the input variable the program got. The program tries to handle multicollinearity with the help of PCA in this case as well.
Both cases the program fit a modell to all independent variables and another where the continuous variables are replaced by using PCA on the them, afterwards automatically evaluate both models and choose the better one, according to Adjusted R-squared, Akaike- and Schwarz information criterias in linear regression. While in the logistic regression analysis the better modell is chosen by accuracy, Akaike- and Schwarz information criterias.

- Categorical variables are estimated with the help of K-Nearest Neighbors method, where k is a hyperparameter, chosen by the analyst, the number of neighbours (values) in an independent variable which determinates the value of the dependent variable as well. In this automated process program search for the optimal number of neighbors by trying out all cases from 1 to 20 neighbours.

**All three cases you can optionally make predictions on new data by giving it's database as the third command-line argument.**

## **Restrictions**
#### Even though this program has some flexibility in its usage let me introduce it's boundaries well.
- Despite the fact that many libraries in python offers functions that can work on their own, I constructed my functions to make my program's usage as automatic as possible thus these functions can work efficiently only in my program and not imported to another file.

- The `data_reader()` function only works with CSV or Excel files so other common source type for instance a SQL database cannot be used as data source.

- In linear regression models heteroskedasticity can cause serious problems if the function format is not suitable. Despite the fact that it can be handled partially by using quadratic term or interactions between variables it is not possible in this case since these variables can't be chosen dinamically(Wooldridge, 2013). Maybe AI integration could mean a solution for this problem to choose the variables.

###Closing thoughts
In spite of it's simplicity my program tries to provide a solution for a relevant consumer need since nowadays employees has to work with a large amount of data even though several cases they dont have the knowledge to analyse it the right way. In the future I want to develop my program to be able to make more sophisticated data cleaning and prediction.

### Sources
- Fisher, R. A. (1936). Iris [Data set]. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/iris

- Hull John. (2020). Machine Learning in Business: An Introduction to the World of Data Science. https://books.google.hu/books/about/Machine_Learning_in_Business.html?id=5uObzQEACAAJ&redir_esc=y

- Hunyadi László & Vita László. (2008). Statisztika I. https://mersz.hu/hunyadi-vita-statisztika-i

- Shoaib Muhammad. (2025). Student Exam Score Dataset Analysis [Data set]. Kaggle. https://www.kaggle.com/datasets/grandmaster07/student-exam-score-dataset-analysis

- Wooldridge Jeffrey. (2013). Introductory econometrics: A modern approach. Mason, OH: South-Western Cengage Learning. http://archive.org/details/introductoryecon0000wool_c3l8
