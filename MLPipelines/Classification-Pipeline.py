import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
# Importing various ML algorithms
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# NOTE: See this link for a description of algorithms: https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/ 
#       SciKit Documentation for algorithms:          https://scikit-learn.org/stable/supervised_learning.html#supervised-learning


# The "answers" to the 20 unknown digits, labeled -1:
answers = [9,9,5,5,6,5,0,9,8,9,8,4,0,1,2,3,4,5,6,7]



def sortData(FILE_NAME):
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
    # Make a list of our models:
    # Realistically, you will want to tune the params of these functions, I am only using the defaults
    models = []
    models.append( ("Logistic Regression",LogisticRegression(solver="liblinear", multi_class="auto")) )
    #models.append( ("Quadratic Discrimination Analysis",QuadraticDiscriminantAnalysis()) )
    #models.append( ("Linear Regression",LinearRegression()) )
    models.append( ("Decision Tree",DecisionTreeClassifier()) )
    models.append( ("Random Forest",RandomForestClassifier()) )
    models.append( ("K Nearest Neighbors",KNeighborsClassifier()) )
    models.append( ("Gaussian Naive Bayes",GaussianNB()) )
    models.append( ("Support Vector Machine",SVC(gamma="scale")) )
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=None)
        cv_results = model_selection.cross_val_score(model, X_known, y_known, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    return

def main():
    FILE_NAME = 'digits5.csv'
    sortData(FILE_NAME)

if __name__ == "__main__":
    main()
