#
# read and model iris data
# 

import numpy as np            
import pandas as pd


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

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df is a "dataframe":
df = pd.read_csv('iris4.csv', header=0)   # read the file w/header row #0

# Now, let's take a look at a bit of the dataframe, df:
df.head()                                 # first five lines
df.info()                                 # column details

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
df['irisname'] = df['irisname'].map(transform)  # apply the function to the column

print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")
print("     +++++ Nearest Neighbors +++++\n\n")

# We first split our dataframe into X (inputs) and y (outputs/targets)
X_all_df = df.iloc[:,0:4]        # iloc == "integer locations" of rows/cols
y_all_df = df[ 'irisname' ]      # individually addressable columns (by name)

# The data is currently in pandas "dataframes," but needs to be in numpy arrays
# These next two lines convert two dataframes to numpy arrays (using .values)
X_all = X_all_df.values        # iloc == "integer locations" of rows/cols
y_all = y_all_df.values      # individually addressable columns (by name)

# This "magic value" of 9 is here because we have looked at the dataset!
X_unlabeled = X_all[:9,:]  # unlabeled up to index 9
y_unlabeled = y_all[:9]    # unlabeled up to index 9

X_labeled_orig = X_all[9:,:]  # labeled data starts at index 9
y_labeled_orig = y_all[9:]    # labeled data starts at index 9



# we scramble the data - but _only_ the labeled data!
# 
indices = np.random.permutation(len(y_labeled_orig))  # indices are a permutation

# we scramble both X and y with the same permutation
X_labeled = X_labeled_orig[indices]              # we apply the same permutation to each!
y_labeled = y_labeled_orig[indices]              # again...





#
# Feature engineering ~ start ~
#


#
# some labels, in case we want to use them...
#
print("Some labels for the graphical tree:")
fn = ['sepallen', 'sepalwid', 'petallen', 'petalwid'] # fn = "feature names"
tn = ['setosa', 'versicolor', 'virginica'] # tn = "target names"


# here is where you can re-scale/change column values...

N = fn.index('sepallen')  # for readability
# X_labeled[:,N] *= 100   # uncomment to make column 0 'sepallen' worth 100x more!

N = fn.index('petalwid')  # readability!
# X_labeled[:,N] *= 100   # uncomment to make column 3 'petalwid' worth 100x more!


#
# Feature engineering ~ end ~
#




#
# We separate into test data and training data ... 
#    + We will train on the training data...
#    + We will test on the testing data and see how well we do!
#    + We don't have a separate validation set of rows; we plan to use cross-validation (below)
TEST_SIZE = 10
X_test = X_labeled[:TEST_SIZE]    # first few are for testing
y_test = y_labeled[:TEST_SIZE]

X_train = X_labeled[TEST_SIZE:]   # all the rest are for training
y_train = y_labeled[TEST_SIZE:]






#
# Create a kNN model and tune its parameters 
# There is only one parameter, k, the number of neighbors considered...
#
from sklearn.neighbors import KNeighborsClassifier
#
# cross-validation
#
# This runs a routine that splits only the training set into two pieces:
# model-building and model-validation. We'll use "build" and "validate"
#

# for k in [1,3,5,7,9,11,15,21,32,42,51,71,91]:
#   knn = KNeighborsClassifier(n_neighbors=k)   # here, k is the "k" in kNN
#   cv_scores = cross_val_score( knn, X_train, y_train, cv=5 ) # cv is the number of splits
#   print('\nthe cv_scores for'+str(k)+'are:')
#   for s in cv_scores:
#       # we format it nicely...
#       s_string = "{0:>#7.4f}".format(s) # docs.python.org/3/library/string.html#formatexamples
#       print("   ",s_string)
#   av = cv_scores.mean()
#   print('+++ with average: ', av)
#   print()



#
# The lab problem (hw4pr1) asks you to create a loop around this cross-validation piece...
# 
# Let k be your loop variable and print all of the results!
#

#
# Here is your choice of k -- this should be the best k found by the loop you create just above
#
best_k = 5


#
# now, train a new model with ALL of the training data...
# using the best value of k, best_k...
# and use this model to predict the labels of the test set
#
# this is a new model! line is where the full training data is used for the model
knn_train = KNeighborsClassifier(n_neighbors=best_k)   # now using the best_k
knn_train.fit(X_train, y_train)                        # using all of the data
print("\nCreated and trained a knn classifier with k =", best_k)  #, knn

# Now, run our test set!
print("For the input data in X_test,")
print("The predicted outputs are")
predicted_labels = knn_train.predict(X_test)
print(predicted_labels)

# and here are the actual labels (iris types)
print("and the actual labels are")
actual_labels = y_test
print(actual_labels)

# 
# let's do more formatted printing!
#

print("\n\n")

#
# formatted printing! (docs.python.org/3/library/string.html#formatstrings)

# the headers
s = "{0:<11} | {1:<11}".format("Predicted","Actual")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)

# the separators
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)

# here is the table...
for pi, ai in zip( predicted_labels, actual_labels ):
    # pi and ai are the index of the predicted and actual label
    plabel = tn[pi]  # tn is target_names - see above
    alabel = tn[ai]
    s = "{0:<11} | {1:<11}".format(plabel,alabel)
    print(s)

print("\n\n")

# 
# here is where you'll do the same thing -- but now for the unlabeled data!
#


###  That is, you'll repeat the use of the above predict-and-check code
###  where this time you should
###  + build a model with ALL the labeled data! Perhaps call it knn_all ...
###  + then, predict the results on the unlabeled data!!
###
###  (the "right answers" are in the hwk description)
###  Note that the "right answers" may not match the model's prediction... .
###  Do include your own labels for the unlabeled data below!
print()
print("Start of prediction of unknown data!")
print()
# Here!!!
knn_train = KNeighborsClassifier(n_neighbors=best_k)   # now using the best_k
knn_train.fit(X_labeled, y_labeled)                        # using all of the data
# Now, run our test set!
print("For the input data in X_unlabeled,")
print("The predicted outputs are")
out = knn_train.predict(X_unlabeled)
print(out)



# import sys   # easy to add break points...
# sys.exit(0)


"""
Comments and results:

Briefly reflect on how this all went...
  + what value of k did you decide on for your kNN?
  + how smoothly did the kNN workflow go...
  + thoughts on this "machine learning pipeline" ?




Also, include the predicted labels of the 9 unlabeled irises!
  + Paste those labels -- or both data and labels -- here:
  + You'll have 9 lines:





"""







#
# an example of how to test inputs you can type in...
#
def test_by_hand(knn):
    """ allows the user to enter values and predict the
        label using the knn model passed in
    """
    print()
    Arr = np.array([[0.,0.,0.,0.]]) # correct-shape array
    T = Arr[0]
    T[0] = float(input("sepal length? "))
    T[1] = float(input("sepal width? "))
    T[2] = float(input("petal length? "))
    T[3] = float(input("petal width? "))
    result = knn.predict(Arr)
    prediction = result[0]
    print("The prediction is", prediction)
    print("result is", result)
    print()


















