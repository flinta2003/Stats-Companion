#***Stats Companion***
###[Please see the visual presentation of my project here!](link)
My program tries to support all those people who are not familiar with the methods of statistical analysises but these methods would be useful for them during their job or any other parts of their life. 

##**Guide**
This program was built to work automatically, to start you have to pass in the name(s) of the database(s). The train database should be the second command-line argument, which helps the program to make predictions on the test database. Test database has to be the third command-line argument. All in all your prompt should look something like this:
`python project.py train_dataset.csv test_dataset.xlsx`

###***Given databeses must only consist of the variable we want to make a prediction for and those variables that used make the predictions!***
For example if we want to predict a student's score on a test:
Train database should contains the score of the test and other independent variables while test database should contain only the independent variables **without the variable you want to predict**.

#https://www.kaggle.com/datasets/grandmaster07/student-exam-score-dataset-analysis
#https://www.kaggle.com/datasets/uciml/iris
```
IDE AZ INDEPENDENT ÉS A DEPENDENT VARIABLE FOGALMÁT BERAKNI EGY TK-BÓL
```

Although it is not necessary to give a test dataset, it can be really helpful when you have to estimate the price of a house according to your database of houses. **It is important to have the same columns in the test dataset as in the train** otherwise the program going to broke.

After that the program will ask automatically whether you would like to |get details | of your dataset. Finally you have the opportunity to choose the statistical method you would like to implement on your data and the program will output the results of the analysis.

##**Functions**
###Data_reader
The `data_reader()` function ensures that the file name(s) which were given in the command-line going to be transformed into pandas dataframes. This function can handle CSV and Excel files with the help of `pd.read_csv` and `pd.read_excel`.


###Cleaning
The cleaning() function has the most functionality all of my functions. 
`cleaning(tanito, X_Var = 0, y_Var = 0, X_test = 0, y_variable = 0, make_dummies = False, y_type = "continuous")`

This function can handle a maximum of 4 datasets `tanito` is the full train database, `X` is all the independent variables while `y` contains only that variable we would like to make a prediction for, finally X_test is the predictive dataset.
Except `tanito` all of them has a default _0_ as during descriptive statistical analysis the program doesn't use them.

With the help of `make_dummies` argument you can choose if you want to make dummies from categorical columns or not. This feature was necessary since this function is also used in the statistical analysis as well.

The function was constructed to be able to handle three types of dependent variables through the `y_type` argument that's default value is "continuous".
**When to use this:** if want to estimate a numberlike variable, _such as the price of a car according to it's features_. 
On the other hand if the variable we would like to predict is binomial or multinomial (*so it has yes/no or categorical values*) it is advised to change the `y_type`'s value.
**When to use "binary" in `y_type`:** in the case of a binomial variable, _for instance when want to estimate whether company will go bankrupt or not_.
**When to use _"categorical"_ in `y_type`:** when the dependent variable has more than two categories, _such as if we want to predict what grade a student will get on his/her test_.


###Descriptive_stat
Responsible for making and presenting the descriptive statistics. It outputs written analysis about the measures, in addition also saves all the statistical measures in a csv file so users can insert it into another report if it is needed.

###Prediction
Prediction function provides the opportunity estimate the value of a variable according to other variables which are in connection with it. With the help of `cleaning()` function it can predict three kinds of variable:
-Continuous variable by using linear regression on all the variables that were given in the input database. The multikollinearity among the independent variables is handled by price component analysis.
-Binary variable by usig logistic regression with all the independent variable the program got. The program tries to handle multikollinearity with the help of PCA in this case as well.
Both cases the program fit a modell to all independent variables and another with using PCA on the continuous variables, after that automatically chose the better model according to Adjusted R-squared, Akaike- and Schwarz information criterias in linear regression. While in the logistic regression analysis the better modell is chosen by accuracy, Akaike- and Schwarz information criterias.
-Categorical variables are estimated with the help of K Nearest Neighbor method, where the K is the number of neighbors (values) in an independent variable which determinates the value of the dependent variable as well. The program search for the optimal number of neighbors by trying out all cases from 1 to 20 neighbors.

**All three cases you can optionally make predictions on new data if it's database is given as a third command-line argument.**

##**Restrictions**
Even though this program has some flexibility in its usage let me introduce it's boundaries well.
-Despite the fact that many libraries in python offers functions that can work on their own, I constructed my functions to make its use as automatic as possible thus it these functions can be used only in my program and not imported to another file.
-In the `data_reader()` function only works with CSV or Excel files so other common source type for instance a SQL database cannot be used to get data from.
-In linear regression models heteroskedasticity can cause serious problems if the function format is not suitable. Despite the fact that it can be handled partially by using quadratic term or interactions between variables it is not possible in this case since these variables can't be chosen dinamically. !!!!!!!!!!Maybe AI integration could mean a solution for this problem.!!!!!!!!!!!!


###Sources
>Hull, J. (2020). Machine Learning in Business: An Introduction to the World of Data Science. https://books.google.hu/books/about/Machine_Learning_in_Business.html?id=5uObzQEACAAJ&redir_esc=y
>Hunyadi, L., & Vita, L. (2008). Statisztika I.
>Wooldridge, J. M. (with Internet Archive). (2013). Introductory econometrics: A modern approach. Mason, OH: South-Western Cengage Learning. http://archive.org/details/introductoryecon0000wool_c3l8




###Closing thoughts
Despite the fact that this program is also able to work now on its own, in the future I would like to make a website for it and present there. In my opinion such a project would be appropiate as a final project of the Cs50 Web Developing course.

In spite of it's simplicity my program tries to provide a solution for a relevant consumer need since nowadays employees has to work with a large amount of data even though they dont have the knowledge to analyse it the right way.