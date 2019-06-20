# Authors: CS-World Domination Summer19 - DM
import numpy as np
import math
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
# Importing various ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def loadAuto(size):
    """Loads data from auto-mpg.csv and gets it into a workable format.
       The size param specifies how much of the data you want split into testing/training"""
    
    df = pd.read_csv('auto.csv', header=0)   # read the file w/header as row 0
    df = df.drop('origin', axis=1)
    df = df.drop('model year', axis=1)
    df = df.drop('car name', axis=1)
    # df.head()
    # df.info()

    # TODO: Repalce this with the data loaded from the other files
    X_unknown = [0]
    y_unknown = [0]
    # Organizing data into training/testing
    # .values converts df to numpy array
    X_known = df.iloc[:,1:].values           # iloc == "integer locations" of rows/cols
    y_known = df[ 'mpg' ].values             # individually addressable columns (by name)

    # It's good practice to scramble your data!
    indices = np.random.permutation(len(X_known))
    X_known = X_known[indices]
    y_known = y_known[indices]

    # Splitting test and training data
    TEST_SIZE = int(len(X_known)*size)
    X_test = X_known[0:TEST_SIZE]
    y_test = y_known[0:TEST_SIZE]
    X_train = X_known[TEST_SIZE:]
    y_train = y_known[TEST_SIZE:]
    return X_known, y_known, X_unknown, y_unknown, X_train, y_train, X_test, y_test

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
    print("The best algorithm after ranking is: "+allList[0][0])
    print("However, do note that the ranking is very basic and does not handle ties...")
    return

def crossValidation(X_known, y_known):
    "Do cross validation tests on your data to help determine the best model and the best params"
    print("\n\n+++ Starting algorithm comparison through cross-validation! +++")
    # Make a list of our models
    # NOTE: Realistically, you will want to tune the params of these functions, I am only using the defaults
    #       You will get warnings by leaving some of the function params empty as the defaults
    models = []
    models.append( ("Linear Regression    ",LinearRegression(fit_intercept=True,normalize=True)))
    # models.append( ("Logistic Regression    ",LogisticRegression(solver="liblinear", multi_class="auto")))
    # models.append( ("Decision Tree          ",DecisionTreeClassifier()) )
    # models.append( ("Random Forest          ",RandomForestClassifier(n_estimators=10)) )

    # Loop through and evaluate each model
    # AccResults = []
    names = []
    allList = []
    # NOTE: If you don't want to bother with confidence intervals, you can just compare the standard deviations
    splits = 10      # If you change the number of splits, you must also change the t-score 
    tscore = 2.262   # Two sided t-score for 95% confidence interval (splits-1 degrees of freedom)
    calc95 = (tscore / math.sqrt(splits))
    # NOTE: See different scoring params: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scoring = ['r2', 'explained_variance']

    print("\nAlgorithm : r2, explained_variance")
    print("*** Results show means for each scoring metric, with 95% Confidence Intervals in parenthesis\n")
    for name, model in models:
        kfold = model_selection.KFold(n_splits=splits, random_state=None)
        cv_scores = model_selection.cross_validate(model, X_known, y_known, cv=kfold, scoring=scoring)
        # AccResults.append(cv_scores["test_accuracy"])
        names.append(name.strip())
        allList.append([name.strip(), cv_scores["test_r2"].mean(), cv_scores["test_explained_variance"].mean()])
        print( "%s: %0.3f (+/- %0.3f)," % (name, cv_scores["test_r2"].mean(), cv_scores["test_r2"].std() * calc95),
        "%.3f (+/- %0.3f)," % (cv_scores["test_explained_variance"].mean(), cv_scores["test_explained_variance"].std() * calc95) )
        # 6"%.3f (+/- %0.3f)" % (cv_scores["test_precision_weighted"].mean(), cv_scores["test_precision_weighted"].std() * calc95) )
    # Function Calls
    # metricRanking(allList)
    # boxPlot(AccResults, names, "Accuracy")
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
    X_train, y_train, X_test, y_test) = loadAuto(0.25)  # Loads the file auto-mpg.csv, and sets important data variables
    visualizeData()                                         # An optional function to be filled out by the user of this code
    crossValidation(X_known, y_known)                       # Comapare different algorithms
    # trainModel(X_train, y_train, X_test, y_test)            # Run the best algorithm on the test/train data
    # predictUnknown(X_known, y_known, X_unknown, y_unknown)  # Run the best algorithm on the unknown data

if __name__ == "__main__":
    main()
