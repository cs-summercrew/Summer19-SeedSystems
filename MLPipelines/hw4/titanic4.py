#
#
# titanic.py
#
#

import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

try: # different imports for different versions of scikit-learn
    from sklearn.model_selection import cross_val_score   # simpler cv this week
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")


# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('titanic4.csv', header=0)
# let's drop columns with too few values or that won't be meaningful
# axis = 1 indicates we want to drop a column, not a row
df = df.drop('body', axis=1)  # The body number (if not survived and body was recovered)
df = df.drop('ticket', axis=1)
df = df.drop('home.dest', axis=1)
df = df.drop('name', axis=1)
df = df.drop('cabin', axis=1)
df = df.drop('embarked', axis=1)
df = df.drop('boat', axis=1)    # If they survived what boat they were on

# let's drop all of the rows with missing data:
df = df.dropna()
# after some data-wrangling, I ended up with 1001 rows (anything over 500-600 seems reasonable)
# You'll need conversion to numeric datatypes for all input columns
#   Here's one example
#
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column

# let's see our dataframe again...
# df.head()
# df.info()
print("+++ end of pandas +++\n")

# separate into input X and target y dataframes...
X_all_df = df.drop('survived', axis=1)        # everything except the 'survival' column
y_all_df = df[ 'survived' ]                   # the target is survival! 
print("+++ start of numpy/scikit-learn +++")
# Note: for the Titanic data, it's the first 42 passengers who are unlabeled
# These next two lines convert two dataframes to numpy arrays (using .values)
X_all = X_all_df.values      
y_all = y_all_df.values      
X_unlabeled = X_all[:42]
y_unlabeled = y_all[:42]
X_labeled = X_all[42:]
y_labeled = y_all[42:]

# Some algorithms are susceptible to being affected by the order of data
# (specifically neural networks) so its good practice to shuffle the data
indices = np.random.permutation(len(y_labeled)) 
X_labeled = X_labeled[indices]
y_labeled = y_labeled[indices]         

# Feature engineering ~ start ~
print("\nFeature Engineering (reweighting)\n")
print("Labels: pclass,sex,age,sibsp,parch,fare")
fn = ["pclass","sex","age","sibsp","parch",'fare']

ind1 = fn.index('pclass')  # for readability
X_labeled[:,ind1] *= 70   # Makes 'pclass' worth 100x more!
ind2 = fn.index('sex')
X_labeled[:,ind2] *= 100
ind3 = fn.index('parch')
X_labeled[:,ind3] *= 6
ind4 = fn.index('sibsp')
X_labeled[:,ind4] *= 3
# Feature engineering ~ end ~

# Splitting test and training data
TEST_SIZE = (len(X_labeled) // 5) # A fifth (20%) of the data is for testing
X_test = X_labeled[:TEST_SIZE]
y_test = y_labeled[:TEST_SIZE]

X_train = X_labeled[TEST_SIZE:]
y_train = y_labeled[TEST_SIZE:]

# Test some loops to find a good k value
if False:
    best = 0
    bestk = 0
    for k in [1,3,5,7,9,11,15,21,32,42,51,71,91]:
        knn = KNeighborsClassifier(n_neighbors=k)   # here, k is the "k" in kNN
        cv_scores = cross_val_score( knn, X_train, y_train, cv=5 ) # cv is the number of splits
        # print('\nthe cv_scores for k = '+str(k)+' are:')
        for s in cv_scores:
            # we format it nicely...
            s_string = "{0:>#7.4f}".format(s) # docs.python.org/3/library/string.html#formatexamples
            # print("   ",s_string)
        av = cv_scores.mean()
        # print('+++ with average: ', av)
        # print()
        if best < av:
            best = av
            bestk = k
    print( "The best is: "+str(bestk)+" with an average of: "+str(best) )
# End of Tests for k value
best_k = 11
# Test the training against the test data
if False:
    knn_train = KNeighborsClassifier(n_neighbors=best_k)
    knn_train.fit(X_train, y_train)
    print("\nCreated and trained a knn classifier with k =", best_k)
    predicted_cats = knn_train.predict(X_test)
    accuracy = knn_train.score(X_test, y_test)
    print("Accuracy: "+str(accuracy))
    print("The predicted categories for X_test are:")
    print(predicted_cats[:10])
    print("The actual categories are:")
    print(y_test[:10])
    # print(predicted_cats == y_test) # shows a bool array of accuracy

# Start of data prediction
if True: 
    knn_train = KNeighborsClassifier(n_neighbors=best_k)
    knn_train.fit(X_labeled, y_labeled)
    print("\nCreated and trained a knn classifier with k =", best_k)
    predicted_cats = knn_train.predict(X_unlabeled)
    print("The predicted categories for the unlabeled data are:")
    print(predicted_cats)
"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?
  + how high were you able to get the average cross-validation (testing) score?
Accuracy: 0.835


Then, include the predicted survival of the unlabeled data (in the original order).
We'll share the known results next week... :-)
[0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0]




"""