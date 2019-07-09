# Authors: CS-World Domination Summer19 - DM
# Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection   # cross_val_score, train_test_split
from sklearn import metrics           # classification_report, confusion_matrix
# ML models Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC     

# Setting Globals for loading Data
# Change one of them to true to load the respective data
# (only one should be true at a time)
titanic = False
iris = False
digits = True

#####################################################################################
########################## Start of load() functions ################################
#####################################################################################

def loadTitanic(size):
    """Loads data from titanic.csv and gets it into a workable format.
       The size param specifies how much of the data you want split into testing/training"""    
    
    df = pd.read_csv('titanic.csv', header=0) # read the file w/header as row 0

    # Drop the useless columns from the data
    df = df.drop('ticket', axis=1)
    df = df.drop('home.dest', axis=1)
    df = df.drop('name', axis=1)
    df = df.drop('cabin', axis=1)
    df = df.drop('embarked', axis=1)

    # One important one is the conversion from string to numeric datatypes!
    # You need to define a function, to help out...
    def tr_mf(s):
        """ from string to number"""
        d = { 'male':0, 'female':1 }
        return d[s]

    df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column
    visualizeData(df)
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

    # It's good practice to scramble/shuffle your data!
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_known, y_known,
    test_size=size, shuffle=True, random_state=None)
    
    return X_known, y_known, X_unknown, y_unknown, X_train, y_train, X_test, y_test

def loadIris(size):
    """Loads data from iris.csv and gets it into a workable format.
       The size param specifies how much of the data you want split into testing/training"""
    
    df = pd.read_csv('iris.csv', header=0)   # read the file w/header as row 0

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
    visualizeData(df)
    X_all = df.iloc[:,0:4].values         # iloc == "integer locations" of rows/cols
    y_all = df[ 'irisname' ].values       # individually addressable columns (by name)

    X_unknown = X_all[:9,:]
    y_unknown = y_all[:9]

    X_known = X_all[9:,:]
    y_known = y_all[9:]

    # It's good practice to scramble/shuffle your data!
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_known, y_known,
    test_size=size, shuffle=True, random_state=None)

    return X_known, y_known, X_unknown, y_unknown, X_train, y_train, X_test, y_test

def loadDigits(size):
    """Loads data from digits.csv and gets it into a workable format.
       The size param specifies how much of the data you want split into testing/training"""
    
    df = pd.read_csv('digits.csv', header=0) # read the file w/header as row 0
    visualizeData(df)
    # Organizing data into training/testing
    # .values converts df to numpy array
    X_all = df.iloc[:,0:64].values           # iloc == "integer locations" of rows/cols
    y_all = df[ '64' ].values                # individually addressable columns (by name)
    X_unknown = X_all[0:20]
    y_unknown = y_all[0:20]
    X_known = X_all[20:]
    y_known = y_all[20:]
    # It's good practice to scramble/shuffle your data!
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_known, y_known,
    test_size=size, shuffle=True, random_state=None)

    return X_known, y_known, X_unknown, y_unknown, X_train, y_train, X_test, y_test

#####################################################################################
########################## End of load() functions ##################################
#####################################################################################

#####################################################################################
######################### Start of main() functions #################################
#####################################################################################

def crossValidation(X_train, y_train):
    """Does cross validation tests on the data to help determine the best model"""
    
    print("\n\n+++ Comparing algorithms with cross-validation! +++")
    # Make a list models to cross-validate
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
    rankList1 = []
    rankList2 = []
    # NOTE: See different scoring params: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scoring = ['accuracy','recall_weighted', 'precision_weighted']
    print("Note that results show the means (of the cross-validating) for each scoring metric, with standard deviation in parenthesis")
    print("\nAlgorithm : Accuracy, Weighted Recall, Weighted Precision")
    for name, model in models:
        # kfold validation
        kfold = model_selection.KFold(n_splits=5, random_state=None)
        cv_scores = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        # Record the results of each metric for printing/visualization
        # NOTE: cross_validate returns dictionaries (cv_scores is a dictionary)
        acc = cv_scores["test_accuracy"]
        F1 = cv_scores["test_recall_weighted"]
        prc = cv_scores["test_precision_weighted"]
        AccResults.append(acc)
        F1Results.append(F1)
        PrcResults.append(prc)
        names.append(name.strip())
        rankList1.append([name.strip(), acc.mean()])
        rankList2.append([name.strip(), prc.mean()])
        print( "%s: %0.3f (+/- %0.3f)," % (name, acc.mean(), acc.std()),
        "%.3f (+/- %0.3f)," % (F1.mean(), F1.std()),
        "%.3f (+/- %0.3f)" % (prc.mean(), prc.std()) )
    # Summarization/Analysis of results
    rankList1 = sorted( rankList1, key=lambda x: x[1], reverse=True )
    rankList2 = sorted( rankList2, key=lambda x: x[1], reverse=True )
    print("The best algorithm after ranking Accuracy is: "+rankList1[0][0])
    print("The best algorithm after ranking Precision is: "+rankList2[0][0])
    if False:
        # The box and whisker plots help show the spread of values from cross-validation better
        # Larger (higher) is better
        boxPlot(AccResults, names, "Accuracy")             
        boxPlot(PrcResults, names, "Precision")
        boxPlot(F1Results, names, "F1 score")
    return

def trainModel(X_train, y_train, X_test, y_test):
    """Run the best model from the cross validation on the test/training data.
       It is a good idea to fine-tune your chosen algorithm in this function."""
    # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
    #       The pipeline function in the link above may help in your fine-tuning
    print("\n+++ Fine-tuning & Summary Data! +++")
    print("Note that the default model is svm, which isn't the best model for the iris or titanic data")
    svc_train = SVC(gamma="scale")
    svc_train.fit(X_train, y_train)
    predictions = svc_train.predict(X_test)
    # Print more summary data for the model
    print("\nConfusion matrix:")
    print(metrics.confusion_matrix(y_test,predictions))
    print("\nClassification report:")
    print(metrics.classification_report(y_test,predictions, digits=3))
    print("The first 30 predicted categories for X_test are:")
    print(predictions[:30])
    print("The first 30 actual categories are:")
    print(y_test[:30])
    return

def predictUnknown(X_known, y_known, X_unknown, y_unknown):
    "Runs the model on the unknown data"
    print("\n\n+++ Predicting unknown data! +++")
    svc_final = KNeighborsClassifier(n_neighbors=5)
    svc_final.fit(X_known, y_known)
    predictions = svc_final.predict(X_unknown)
    print("\nThe predicted categories vs actual:")
    print(predictions)
    # Load the correct answers to the unknown data
    global titanic
    global iris
    global digits
    if digits: 
        # The "answers" for digits.csv:
        answers = [9,9,5,5,6,5,0,9,8,9,8,4,0,1,2,3,4,5,6,7]
    if iris: 
        # The "answers" for iris.csv:
        answers = [2,2,1,1,0,0,2,1,0]
    if titanic:
        # The "answers" for titanic.csv:
        answers = [0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,1,1,1,0,0,0,1,1,0,1,0]
    print(np.array(answers))
    accuracy = svc_final.score(X_unknown, answers)
    print("Accuracy: "+str(accuracy))
    return

#####################################################################################
########################## End of main() functions ##################################
#####################################################################################

#####################################################################################
######################### Start of helper functions #################################
#####################################################################################

def visualizeData(df):
    """It is often a good idea to visualize data before starting to working with it."""
    print("\n+++ Visualizing the feature data! +++")
    if True:
        df.hist()
    if False:
        from pandas.plotting import scatter_matrix
        scatter_matrix(df)
    if False: 
        df.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
    plt.show()
    return

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
    return

#####################################################################################
########################## End of helper functions ##################################
#####################################################################################

def main():

    # Data Pre-processing
    # NOTE: Try loading different csv files. SVM was the best algorithm for digits.csv, and is used as the default
    #       example for testing the iris and titanic data. For the load functions, generally use a value from 0.10-0.30
    global titanic
    global iris
    global digits
    if iris:
        (X_known, y_known, X_unknown, y_unknown,
        X_train, y_train, X_test, y_test) = loadIris(0.25)    # Loads the file iris.csv, and sets important data variables
    if titanic:
        (X_known, y_known, X_unknown, y_unknown,
        X_train, y_train, X_test, y_test) = loadTitanic(0.20) # Loads the file titanic.csv, and sets important data variables
    if digits:
        (X_known, y_known, X_unknown, y_unknown,
        X_train, y_train, X_test, y_test) = loadDigits(0.15)  # Loads the file digits.csv, and sets important data variables
    
    # Model Selection/Refinement
    crossValidation(X_train, y_train)                       # Compare different algorithms
    trainModel(X_train, y_train, X_test, y_test)            # Run the best algorithm on the test/train data
    # Test prediction on unknown data
    predictUnknown(X_known, y_known, X_unknown, y_unknown) 

if __name__ == "__main__":
    main()
