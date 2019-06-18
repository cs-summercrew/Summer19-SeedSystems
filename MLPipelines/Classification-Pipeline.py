import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
# Importing various ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# NOTE: See this link for a description of algorithms: https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/ 
#       SciKit Documentation for algorithms:          https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

# NOTE: Partially based off of the following tutorial: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

# The "answers" to the 20 unknown digits, labeled -1:
answers = [9,9,5,5,6,5,0,9,8,9,8,4,0,1,2,3,4,5,6,7]


def boxPlot(results, names):
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def crossValidation(X_known, y_known):
    # Make a list of our models:
    # NOTE: Realistically, you will want to tune the params of these functions, I am only using the defaults
    #       You will get warnings by leaving some of these functions' params entirely as the defaults
    models = []
    models.append( ("Logistic Regression    ",LogisticRegression(solver="liblinear", multi_class="auto")))
    models.append( ("Decision Tree          ",DecisionTreeClassifier()) )
    models.append( ("Random Forest          ",RandomForestClassifier(n_estimators=10)) )
    models.append( ("K Nearest Neighbors    ",KNeighborsClassifier()) )
    models.append( ("Gaussian Naive Bayes   ",GaussianNB()) )
    models.append( ("Support Vector Machine ",SVC(gamma="scale")) )

    # evaluate each model in turn
    AccResults = []
    PrcResults = []
    names = []
    # NOTE: See different scoring params: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scoring = ['accuracy','f1_weighted', 'precision_weighted']
    # BUG: Check that I am using the right type of scoring params
    #      Also check the confidence interval calculation
    
    print("\nAlgorithm : Accuracy, Weighted f1 score, Weighted Precision")
    print("*** Results show means for each scoring metric, with 95% Confidence Intervals in parenthesis\n")    
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=None)
        cv_scores = model_selection.cross_validate(model, X_known, y_known, cv=kfold, scoring=scoring)
        AccResults.append(cv_scores["test_accuracy"])
        PrcResults.append(cv_scores["test_precision_weighted"])
        names.append(name)
        print( "%s: %0.3f (+/- %0.3f)," % (name, cv_scores["test_accuracy"].mean(), cv_scores["test_accuracy"].std() * 2),
        "%.3f (+/- %0.3f)," % (cv_scores["test_f1_weighted"].mean(), cv_scores["test_f1_weighted"].std() * 2),
        "%.3f (+/- %0.3f)" % (cv_scores["test_precision_weighted"].mean(), cv_scores["test_precision_weighted"].std() * 2) )
    boxPlot(AccResults, names)
    boxPlot(PrcResults, names)
    return

def main():
    FILE_NAME = 'digits5.csv'
    df = pd.read_csv(FILE_NAME, header=0)    # read the file w/header as row 0
    # df.head()                                  # first five lines
    # df.info()                                  # column details

    # Organizing data into training/testing
    # .values converts df to numpy array
    X_all = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
    y_all = df[ '64' ].values             # individually addressable columns (by name)
    X_unknown = X_all[:20]
    y_unknown = y_all[:20]
    X_known = X_all[20:]
    y_known = y_all[20:]
    # It is good practice to scramble your data!
    indices = np.random.permutation(len(X_known))
    X_known = X_known[indices]
    y_known = y_known[indices]

    # Splitting test and training data
    TEST_SIZE = (len(X_known) // 5) # A fifth (20%) of the data is for testing
    X_test = X_known[:TEST_SIZE]
    y_test = y_known[:TEST_SIZE]
    X_train = X_known[TEST_SIZE:]
    y_train = y_known[TEST_SIZE:]

    # Function Calls
    crossValidation(X_known, y_known)

if __name__ == "__main__":
    main()
