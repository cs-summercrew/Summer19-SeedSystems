#
# titanic5: modeling the Titanic data with DTs and RFs
#

import numpy as np            
import pandas as pd

from sklearn import tree      # for decision trees
from sklearn import ensemble  # for random forests

try: # different imports for different versions of scikit-learn
    from sklearn.model_selection import cross_val_score   # simpler cv this week
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")


#
# The "answers" to the 30 unlabeled passengers:
#
answers = [0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,
            1,0,1,1,1,1,0,0,0,1,1,0,1,0]
answers = np.array(answers)
#

print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('titanic5.csv', header=0)    # read the file w/header row #0
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
#
print("\n+++ End of pandas +++\n")

#

print("+++ Start of numpy/scikit-learn +++\n")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays 
X_all = df.drop('survived', axis=1).values       
y_all = df[ 'survived' ].values      
X_unlabeled = X_all[:30]
y_unlabeled = y_all[:30]
X_labeled = X_all[30:]
y_labeled = y_all[30:]
# we can scramble the data - but only the labeled data!
indices = np.random.permutation(len(X_labeled))
X_train = X_labeled[indices]
y_train = y_labeled[indices]

fn = ["pclass","sex","age","sibsp","parch",'fare']
tn = ["died","survived"]
#
# now, building from iris5.py and/or digits5.py
#      create DT and RF models on the Titanic dataset!
#
#      Goal: find feature importances ("explanations")
#      Challenge: can you get over 80% CV accuracy?
# 

# cross-validation and scoring to determine parameter: max_depth
if False:
    best_depth = 0
    best_av = 0
    for max_depth in range(1,50):
        # create our classifier
        dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
        #
        # cross-validate to tune our model (this week, all-at-once)
        #
        scores = cross_val_score(dtree, X_train, y_train, cv=5)
        average_cv_score = scores.mean()
        if average_cv_score > best_av:
            best_av = average_cv_score
            best_depth = max_depth
    print("For the best depth of",best_depth, "the average CV score = ", best_av)

# choose a MAX_DEPTH based on cross-validation... 
MAX_DEPTH = 3
# Predict with our DT model
if True:
    print()
    print("Using MAX_DEPTH=", MAX_DEPTH)
    dtree = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
    dtree = dtree.fit(X_train, y_train)
    print("Decision-tree predictions:\n")
    predicted_labels = dtree.predict(X_unlabeled)
    print(predicted_labels)
    print(answers)
    print(predicted_labels == answers)
    accuracy = dtree.score(X_unlabeled, answers)
    print("\nAccuracy: "+str(accuracy))
    

# cross-validation and scoring to determine parameter: NUM_TREES
if False:
    best_numt = 0
    best_av = 0
    for numt in range(250,300):
        print(numt)
        rforest = ensemble.RandomForestClassifier(max_depth=3, n_estimators=numt)
        scores = cross_val_score(rforest, X_train, y_train, cv=5)
        average_cv_score = scores.mean()
        if average_cv_score > best_av:
            best_av = average_cv_score
            best_numt = numt
    print("Best numt is",best_numt, "the average CV score = ", best_av)
# choose values based on cross-validation... 
NUM_TREES = 270
MAX_DEPTH = 8
# Predict with our RT model
if True:
    print()
    print("Using MAX_DEPTH=", MAX_DEPTH, "and NUM_TREES=", NUM_TREES)
    rforest = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
    rforest = rforest.fit(X_train, y_train)
    print(X_train.shape)
    # here are some examples, printed out:
    print("\nRandom-forest predictions:\n")
    predicted_labels = rforest.predict(X_unlabeled)
    print(predicted_labels)
    print(answers)
    print(predicted_labels == answers)
    accuracy = rforest.score(X_unlabeled, answers)
    print("\nAccuracy: "+str(accuracy))
    



