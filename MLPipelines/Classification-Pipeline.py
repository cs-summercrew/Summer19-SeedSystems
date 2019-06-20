# Authors: CS-World Domination Summer19 - DM
import numpy as np
import math
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import classification_report,confusion_matrix
# Importing various ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# NOTE: See this link for a description of algorithms: https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/ 
#       SciKit Documentation of its algorithms:        https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

def loadTitanic(size):
    """Loads data from titanic.csv and gets it into a workable format.
       The size param specifies how much of the data you want split into testing/training"""    
    
    df = pd.read_csv('titanic.csv', header=0) # read the file w/header as row 0
    # df.head()                               # first five lines
    # df.info()                               # column details

    # Drop the useless columns from the data
    df = df.drop('ticket', axis=1)
    df = df.drop('home.dest', axis=1)
    df = df.drop('name', axis=1)
    df = df.drop('cabin', axis=1)
    df = df.drop('embarked', axis=1)

    # One important one is the conversion from string to numeric datatypes!
    # You need to define a function, to help out...
    def tr_mf(s):
        """ from string to number
        """
        d = { 'male':0, 'female':1 }
        return d[s]

    df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column
    #
    # end of conversion to numeric data...
    # drop rows with missing data!
    df = df.dropna()

    # Organizing data into training/testing
    # .values converts df to numpy array
    X_all = df.drop('survived', axis=1).values       # iloc == "integer locations" of rows/cols
    y_all = df[ 'survived' ].values                  # individually addressable columns (by name)
    X_unknown = X_all[:30]
    y_unknown = y_all[:30]
    X_known = X_all[30:]
    y_known = y_all[30:]

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

def loadIris(size):
    """Loads data from iris.csv and gets it into a workable format.
       The size param specifies how much of the data you want split into testing/training"""
    
    df = pd.read_csv('iris.csv', header=0)   # read the file w/header as row 0
    # df.head()                              # first five lines
    # df.info()                              # column details

    def transform(s):
        """ from string to number
            setosa -> 0
            versicolor -> 1
            virginica -> 2
            unknown -> -1
        """
        d = { 'unknown':-1, 'setosa':0, 'versicolor':1, 'virginica':2 }
        return d[s]
    df['irisname'] = df['irisname'].map(transform)  # apply the function to the column

    X_all = df.iloc[:,0:4].values         # iloc == "integer locations" of rows/cols
    y_all = df[ 'irisname' ].values       # individually addressable columns (by name)

    X_unknown = X_all[:9,:]
    y_unknown = y_all[:9]

    X_known = X_all[9:,:]
    y_known = y_all[9:]
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

def loadDigits(size):
    """Loads data from digits.csv and gets it into a workable format.
       The size param specifies how much of the data you want split into testing/training"""
    
    df = pd.read_csv('digits.csv', header=0) # read the file w/header as row 0
    # df.head()                              # first five lines
    # df.info()                              # column details

    # Organizing data into training/testing
    # .values converts df to numpy array
    X_all = df.iloc[:,0:64].values           # iloc == "integer locations" of rows/cols
    y_all = df[ '64' ].values                # individually addressable columns (by name)
    X_unknown = X_all[0:20]
    y_unknown = y_all[0:20]
    X_known = X_all[20:]
    y_known = y_all[20:]
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
    models.append( ("Logistic Regression    ",LogisticRegression(solver="liblinear", multi_class="auto")))
    models.append( ("Decision Tree          ",DecisionTreeClassifier()) )
    models.append( ("Random Forest          ",RandomForestClassifier(n_estimators=10)) )
    models.append( ("K Nearest Neighbors    ",KNeighborsClassifier()) )
    models.append( ("Gaussian Naive Bayes   ",GaussianNB()) )
    models.append( ("Support Vector Machine ",SVC(gamma="scale")) )

    # Loop through and evaluate each model
    AccResults = []
    PrcResults = []
    F1Results = []
    names = []
    allList = []
    # NOTE: If you don't want to bother with confidence intervals, you can just compare the standard deviations
    splits = 10      # If you change the number of splits, you must also change the t-score 
    tscore = 2.262   # Two sided t-score for 95% confidence interval (splits-1 degrees of freedom)
    calc95 = (tscore / math.sqrt(splits))
    # NOTE: See different scoring params: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scoring = ['accuracy','f1_weighted', 'precision_weighted']
    # BUG: Unsure if I am using the right type of scoring params (was getting errors using the non-weighted for digits data)
    print("\nAlgorithm : Accuracy, Weighted f1 score, Weighted Precision")
    print("*** Results show means for each scoring metric, with 95% Confidence Intervals in parenthesis\n")
    for name, model in models:
        kfold = model_selection.KFold(n_splits=splits, random_state=None)
        cv_scores = model_selection.cross_validate(model, X_known, y_known, cv=kfold, scoring=scoring)
        AccResults.append(cv_scores["test_accuracy"])
        PrcResults.append(cv_scores["test_precision_weighted"])
        F1Results.append(cv_scores["test_f1_weighted"])
        names.append(name.strip())
        allList.append([name.strip(), cv_scores["test_accuracy"].mean(), cv_scores["test_f1_weighted"].mean(), cv_scores["test_precision_weighted"].mean()])
        print( "%s: %0.3f (+/- %0.3f)," % (name, cv_scores["test_accuracy"].mean(), cv_scores["test_accuracy"].std() * calc95),
        "%.3f (+/- %0.3f)," % (cv_scores["test_f1_weighted"].mean(), cv_scores["test_f1_weighted"].std() * calc95),
        "%.3f (+/- %0.3f)" % (cv_scores["test_precision_weighted"].mean(), cv_scores["test_precision_weighted"].std() * calc95) )
    # Function Calls
    metricRanking(allList)
    boxPlot(AccResults, names, "Accuracy")
    boxPlot(PrcResults, names, "Precision")
    boxPlot(F1Results, names, "F1 score")
    return

def trainModel(X_train, y_train, X_test, y_test):
    """Run the best model from the cross validation on the test/training data.
       It is a good idea to fine-tune your chosen algorithm in this function."""
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
    #       The pipeline function in the link above may help in your fine-tuning
    print("\n\n+++ Starting fine-tuning with svm! +++")
    svc_train = SVC(gamma="scale")
    svc_train.fit(X_train, y_train)
    predictions = svc_train.predict(X_test)
    # Print more summary data for the model
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test,predictions))
    print("\nClassification report:")
    print(classification_report(y_test,predictions, digits=3))
    print("The first 30 predicted categories for X_test are:")
    print(predictions[:30])
    print("The first 30 actual categories are:")
    print(y_test[:30])
    return

def predictUnknown(X_known, y_known, X_unknown, y_unknown):
    "Runs the model on the unknown data"
    print("\n\n+++ Starting the prediction of unknown data! +++")
    svc_final = KNeighborsClassifier(n_neighbors=5)
    svc_final.fit(X_known, y_known)
    predictions = svc_final.predict(X_unknown)
    print("\nThe predicted categories vs actual:")
    print(predictions)
    
    try:
        # The "answers" for digits.csv:
        answers = [9,9,5,5,6,5,0,9,8,9,8,4,0,1,2,3,4,5,6,7]
        # The "answers" for iris.csv:
        # answers = [2,2,1,1,0,0,2,1,0]
        # The "answers" for titanic.csv:
        # answers = [0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,
        #         1,0,1,1,1,1,0,0,0,1,1,0,1,0]
        print(np.array(answers))
        accuracy = svc_final.score(X_unknown, answers)
        print("Accuracy: "+str(accuracy))
    except:
        print("You didn't uncomment the correct answers list for the file you're reading!")
    
    return


def main():

    # NOTE: Uncomment to load different csv files. SVM was the best algorithm for digits.csv, and is used as the default
    #       example for testing the iris and titanic data. For the load functions, generally use a value from 0.10-0.30

    # (X_known, y_known, X_unknown, y_unknown,
    # X_train, y_train, X_test, y_test) = loadIris(0.25)    # Loads the file iris.csv, and sets important data variables

    # (X_known, y_known, X_unknown, y_unknown,
    # X_train, y_train, X_test, y_test) = loadTitanic(0.20) # Loads the file titanic.csv, and sets important data variables

    (X_known, y_known, X_unknown, y_unknown,
    X_train, y_train, X_test, y_test) = loadDigits(0.25)  # Loads the file digits.csv, and sets important data variables
    
    visualizeData()                                         # An optional function to be filled out by the user of this code
    crossValidation(X_known, y_known)                       # Comapare different algorithms
    trainModel(X_train, y_train, X_test, y_test)            # Run the best algorithm on the test/train data
    predictUnknown(X_known, y_known, X_unknown, y_unknown)  # Run the best algorithm on the unknown data

if __name__ == "__main__":
    main()
