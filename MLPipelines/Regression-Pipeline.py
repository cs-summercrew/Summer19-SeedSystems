# Authors: CS-World Domination Summer19 - DM
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import utils
from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
# Importing various ML algorithms
from sklearn import metrics, svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lars
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import TheilSenRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def loadData(size):
    """Loads data from a csv and gets it into a workable format.
       The size param specifies how much of the data you want split into testing/training"""
    
    df = pd.read_csv('auto.csv', header=0)   # read the file w/header as row 0
    df = df.drop('origin', axis=1)
    df = df.drop('model year', axis=1)
    df = df.drop('car name', axis=1)                          # column details
    # df.head()
    # df.info()
    # print(df.describe())

    # TODO: Replace this with the data loaded from the other files
    X_unknown = [0]
    y_unknown = [0]
    # Organizing data into training/testing
    # .values converts df to numpy array
    X_known = df.iloc[:,1:].values           # iloc == "integer locations" of rows/cols
    y_known = df[ 'mpg' ].values             # individually addressable columns (by name)

    # It's good practice to scramble/shuffle your data!
    X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=size, shuffle=True, random_state=None)

    # For many algorithms, inputs/outputs must be categorical (ints not floats)
    # https://stackoverflow.com/questions/41925157/logisticregression-unknown-label-type-continuous-using-sklearn-in-python
    # print(y_train)
    # lab_enc = preprocessing.LabelEncoder()
    # encoded = lab_enc.fit_transform(y_train)
    # print(encoded)

    return X_known, y_known, X_unknown, y_unknown, X_train, y_train, X_test, y_test

def scaleData(X_train, X_test):
    # https://www.kaggle.com/discdiver/guide-to-scaling-and-standardizing
    mm_scaler = preprocessing.MinMaxScaler()
    X_train = mm_scaler.fit_transform(X_train)
    X_test = mm_scaler.fit_transform(X_test)

    return X_train, X_test

def visualizeData():
    "It is often a good idea to visualize your data before you start working with it"
    pass

def boxPlot(results, names, metric):
    """ This box plot shows the spread of the data, NOT the confidence interval!!! 
        The box extends from the lower to upper quartile values of the data, with a line at the median. 
        The whiskers extend from the box to show the range of the data. """
    fig = plt.figure()
    fig.suptitle('Algorithm '+metric+' Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def metricRanking(allList):
    "Assigns a ranking based on each score, prints the name of the metric with the (best) lowest cumulative ranking"
    print()
    # Rank the metric performance for each algorithm
    # First to be ranked is accuracy
    allList = sorted( allList, key=lambda x: x[1], reverse=True )
    for i in range(len(allList)):
        allList[i][1] = i+1
    # Second to be ranked is precision
    allList = sorted( allList, key=lambda x: x[2], reverse=True )
    for i in range(len(allList)):
        allList[i][2] = i+1
    # Third to be ranked is f1 score
    allList = sorted( allList, key=lambda x: x[3], reverse=True )
    for i in range(len(allList)):
        allList[i][3] = i+1
    # Combine the scores of each metric for each algorithm
    # The best algorithm is the one with the lowest ranking
    for i in range(len(allList)):
        cumulative_score = allList[i][1] + allList[i][2] + allList[i][3]
        allList[i] = [allList[i][0]]
        allList[i].append(cumulative_score)
    allList = sorted( allList, key=lambda x: x[1], reverse=False)
    print("The best algorithm after ranking is: "+allList[0][0])
    print(allList)
    print("However, do note that the ranking is very basic and does not handle ties...")
    return

def crossValidation(X_train, y_train):
    "Do cross validation tests on your data to help determine the best model and the best params"
    print("\n\n+++ Starting algorithm comparison through cross-validation! +++")
    # Make a list of our models
    # NOTE: Realistically, you will want to tune the params of these functions, I am only using the defaults
    #       You will get warnings by leaving some of the function params empty as the defaults
    models = []
    models.append( ("OLS                ",LinearRegression()) )
    models.append( ("SVR                ",svm.SVR(gamma="scale")) )
    models.append( ("BayesianRidge      ",BayesianRidge()) )
    models.append( ("ARD                ",ARDRegression()) )
    models.append( ("TheilSen           ",TheilSenRegressor()) )
    models.append( ("Lars               ",Lars()) )
    models.append( ("PassiveAggressive  ",PassiveAggressiveRegressor()) )
    models.append( ("SGD                ",SGDRegressor()) )

    # Loop through and evaluate each model
    r2Results = []
    maeResults = []
    mseResults = []
    names = []
    allList = []
    # NOTE: If you don't want to bother with confidence intervals, you can just compare the standard deviations
    splits = 10
    tscore = 2.262
    calc95 = (tscore / math.sqrt(splits))
    # NOTE: See different scoring params: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scoring = ['r2','neg_mean_absolute_error', 'neg_mean_squared_error']
    print("\nAlgorithm : R Squared, Mean Absolute Error, Mean Standard Error")
    print("*** Results show means for each scoring metric, with 95% Confidence Intervals in parenthesis\n")
    for name, model in models:
        kfold = model_selection.KFold(n_splits=splits, random_state=None)
        cv_scores = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        r2Results.append(cv_scores["test_r2"])
        maeResults.append(cv_scores["test_neg_mean_absolute_error"])
        mseResults.append(cv_scores["test_neg_mean_squared_error"])
        names.append(name.strip())
        allList.append([name.strip(), cv_scores["test_r2"].mean(), cv_scores["test_neg_mean_absolute_error"].mean(), cv_scores["test_neg_mean_squared_error"].mean()])
        print( "%s: %0.3f (+/- %0.3f)," % (name, cv_scores["test_r2"].mean(), cv_scores["test_r2"].std() * calc95),
        "%.3f (+/- %0.3f)," % (cv_scores["test_neg_mean_absolute_error"].mean(), cv_scores["test_neg_mean_absolute_error"].std() * calc95),
        "%.3f (+/- %0.3f)" % (cv_scores["test_neg_mean_squared_error"].mean(), cv_scores["test_neg_mean_squared_error"].std() * calc95) )
    # Function Calls
    # metricRanking(allList)
    boxPlot(r2Results, names, "R Squared")
    boxPlot(maeResults, names, "Mean Absolute Error")
    boxPlot(mseResults, names, "Mean Squared Error")
    return

def trainModel(X_train, y_train, X_test, y_test):
    """Run the best model from the cross validation on the test/training data.
       It is a good idea to fine-tune your chosen algorithm in this function."""
    return

def predictUnknown(X_known, y_known, X_unknown, y_unknown):
    "Runs the model on the unknown data"
    return


def main():
    (X_known, y_known, X_unknown, y_unknown,
    X_train, y_train, X_test, y_test) = loadData(0.20)  # Loads the csv file, and sets important data variables

    (X_train, X_test) = scaleData(X_train, X_test)

    visualizeData()                                         # An optional function to be filled out by the user of this code
    crossValidation(X_train, y_train)                       # Compare different algorithms
    # trainModel(X_train, y_train, X_test, y_test)            # Run the best algorithm on the test/train data
    # predictUnknown(X_known, y_known, X_unknown, y_unknown)  # Run the best algorithm on the unknown data

if __name__ == "__main__":
    main()
