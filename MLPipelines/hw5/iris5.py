#
# read iris data
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
# Here are the correct answers to the csv's "unknown" flowers
#
answers = [ 'virginica',   # index 0 (row 1 in the csv) 
            'virginica',   # index 1 (row 2 in the csv) 
            'versicolor',  # and so on...
            'versicolor',
            'setosa',
            'setosa',
            'virginica',
            'versicolor',
            'setosa']



print("+++ Start of pandas' datahandling +++\n")

# df is a "dataframe":
df = pd.read_csv('iris5.csv', header=0)   # read the file w/header row #0

# Now, let's take a look at a bit of the dataframe, df:
# df.head()                                 # first five lines
# df.info()                                 # column details

# One important feature is the conversion from string to numeric datatypes!
# For _input_ features, numpy and scikit-learn need numeric datatypes
# You can define a transformation function, to help out...
def transform(s):
    """ from string to number
          setosa -> 0
          versicolor -> 1
          virginica -> 2
    """
    d = { 'unknown':-1, 'setosa':0, 'versicolor':1, 'virginica':2 }
    return d[s]
    
# 
# this applies the function transform to a whole column
#
# df['irisname'] = df['irisname'].map(transform)  # apply the function to the column

print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")

print("     +++++ Decision Trees +++++\n\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_all = df.iloc[:,0:4].values        # iloc == "integer locations" of rows/cols
y_all = df[ 'irisname' ].values      # individually addressable columns (by name)

X_unlabeled = X_all[:9,:]  # the "unknown" flower species (see above!)
y_unlabeled = y_all[:9]    # these are "unknown"

X_labeled = X_all[9:,:]  # make the 10 into 0 to keep all of the data
y_labeled = y_all[9:]    # same for this line

#
# we can scramble the data - but only the labeled data!
# 
indices = np.random.permutation(len(X_labeled))  # this scrambles the data each time
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

#
# Notice that, here, we will _only_ use cross-validation for model-buidling
#   (We won't include a separate X_train X_test split.)
#

X_train = X_data_full
y_train = y_data_full

#
# some labels to make the graphical trees more readable...
#
print("Some labels for the graphical tree:")
feature_names = ['sepallen', 'sepalwid', 'petallen', 'petalwid']
target_names = ['setosa', 'versicolor', 'virginica']

#
# show the creation of three tree files (at three max_depths)
#
if False:
    for max_depth in [1,2,3]:
        # the DT classifier
        dtree = tree.DecisionTreeClassifier(max_depth=max_depth)

        # train it (build the tree)
        dtree = dtree.fit(X_train, y_train) 

        # write out the dtree to tree.dot (or another filename of your choosing...)
        filename = 'tree' + str(max_depth) + '.dot'
        tree.export_graphviz(dtree, out_file=filename,   # the filename constructed above...!
                                feature_names=feature_names,  filled=True, 
                                rotate=False, # True for Left-to-Right; False for Up-Down
                                class_names=target_names, 
                                leaves_parallel=True )  # lots of options!
        #
        # Visualize the resulting graphs (the trees) at www.webgraphviz.com
        #
        print("Wrote the file", filename)  
        #



#
# cross-validation and scoring to determine parameter: max_depth
# 
if False:
    for max_depth in range(1,12):
        # create our classifier
        dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
        #
        # cross-validate to tune our model (this week, all-at-once)
        #
        scores = cross_val_score(dtree, X_train, y_train, cv=5)
        average_cv_score = scores.mean()
        print("      Scores:", scores)
        print("For depth=", max_depth, "average CV score = ", average_cv_score)  
    

# import sys
# print("bye!")
# sys.exit(0)



MAX_DEPTH = 3   # choose a MAX_DEPTH based on cross-validation... 
print("\nChoosing MAX_DEPTH =", MAX_DEPTH, "\n")

#
# now, train the model with ALL of the training data...  and predict the unknown labels
#


# our decision-tree classifier...
dtree = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
dtree = dtree.fit(X_train, y_train) 

#
# and... Predict the unknown data labels
#
print("Decision-tree predictions:\n")
predicted_labels = dtree.predict(X_unlabeled)
answer_labels = answers

#
# formatted printing! (docs.python.org/3/library/string.html#formatstrings)
#
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table...
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

#
# feature importances!
#
print()
print("dtree.feature_importances_ are\n      ", dtree.feature_importances_) 
print("Order:", feature_names[0:4])


#
# now, show off Random Forests!
# 

print("\n\n")
print("     +++++ Random Forests +++++\n\n")

#
# The data is already in good shape -- let's start from the original dataframe:
#
X_all = df.iloc[:,0:4].values        # iloc == "integer locations" of rows/cols
y_all = df[ 'irisname' ].values      # individually addressable columns (by name)

#
# Labeled and unlabeled data...
#

X_unlabeled = X_all[:9,:]  # the "unknown" flower species (see above!)
y_unlabeled = y_all[:9]    # these are "unknown"

X_labeled = X_all[9:,:]  # just the input features, X, that HAVE output labels
y_labeled = y_all[9:]    # here are the output labels, y, for X_labeled

#
# we can scramble the data - but only the labeled data!
# 
indices = np.random.permutation(len(X_labeled))  # this scrambles the data each time
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

X_train = X_data_full
y_train = y_data_full

#
# Again, we will use cross-validation to determine the Random Forest's two hyperparameters:
#   + max_depth
#   + n_estimators
#
# (We will not have an X_train vs. X_test split in addition.)
#

#
# Lab task!  Your goal:
#   + loop over a number of values of max_depth (m)
#   + loop over different numbers of trees/n_estimators (n)
#   -> to find a pair of values that results in a good average CV score
#
# use the decision-tree code above as a template for this...
#

#
# You need to "loopify" the code below...
#
if False:
    # here is a _single_ example call to build a RF:
    m = 2
    n = 10
    rforest = ensemble.RandomForestClassifier(max_depth=2, n_estimators=10)

    # an example call to run 5x cross-validation on the labeled data
    scores = cross_val_score(rforest, X_train, y_train, cv=5)
    print("CV scores:", scores)
    print("CV scores' average:", scores.mean())

# you'll want to take the average of these...



#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#


# these next lines is where the full training data is used for the model
MAX_DEPTH = 4
NUM_TREES = 250
print()
print("Using MAX_DEPTH=", MAX_DEPTH, "and NUM_TREES=", NUM_TREES)
rforest = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
rforest = rforest.fit(X_train, y_train) 
acc = rforest.score(X_unlabeled, answers)
print("Accuracy: ", acc)
# here are some examples, printed out:
print("Random-forest predictions:\n")
predicted_labels = rforest.predict(X_unlabeled)
answer_labels = answers  # because we know the answers, above!

#
# formatted printing again (see above for reference link)
#
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table...
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

#
# feature importances
#
print("\nrforest.feature_importances_ are\n      ", rforest.feature_importances_) 
print("Order:", feature_names[0:4])

# The individual trees are in  rforest.estimators_  [a list of decision trees!]
