#
#
# digits4.py
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
df = pd.read_csv('digits4.csv', header=0)
# df.head()
# df.info()

# Convert feature columns as needed...
# You may to define a function, to help out:
def transform(s):
    """ from number to string
    """
    return 'digit ' + str(s)
    
df['label'] = df['64'].map(transform)  # the label in column 64
print("+++ End of pandas +++\n")

# import sys
# sys.exit(0)

# separate the data into input X and target y dataframes...
X_all_df = df.drop('label', axis=1)        # everything except the 'label' column
y_all_df = df[ 'label' ]                   # the label is the target! 

print("+++ start of numpy/scikit-learn +++")

# The data is currently in pandas "dataframes," but needs to be in numpy arrays
# These next two lines convert two dataframes to numpy arrays (using .values)
X_all = X_all_df.values        # iloc == "integer locations" of rows/cols
y_all = y_all_df.values      # individually addressable columns (by name)

X_unlabeled_partial = X_all[0:10]
y_unlabeled_partial = y_all[0:10]

X_unlabeled_full = X_all[10:22]
y_unlabeled_full = y_all[10:22]

X_labeled = X_all[22:]# 1797
y_labeled = y_all[22:]# 1797

# Some algorithms are susceptible to being affected by the order of data
# (specifically neural networks) so its good practice to shuffle the data
indices = np.random.permutation(len(y_labeled)) 
X_labeled = X_labeled[indices]              # we apply the same permutation to each!
y_labeled = y_labeled[indices]              # again...

# Feature engineering ~ start ~
print("\nFeature Engineering (reweighting)\n")
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
        print('\nthe cv_scores for k = '+str(k)+' are:')
        for s in cv_scores:
            # we format it nicely...
            s_string = "{0:>#7.4f}".format(s) # docs.python.org/3/library/string.html#formatexamples
            print("   ",s_string)
        av = cv_scores.mean()
        print('+++ with average: ', av)
        print()
        if best < av:
            best = av
            bestk = k
    print( "The best is: "+str(bestk)+" with an average of: "+str(best) )
# End of Tests for k value
best_k = 1
# Test the training against the test data
if False:
    knn_train = KNeighborsClassifier(n_neighbors=best_k)
    knn_train.fit(X_train, y_train)
    print("\nCreated and trained a knn classifier with k =", best_k)
    predicted_cats = knn_train.predict(X_test)
    accuracy = knn_train.score(X_test, y_test)
    print("Accuracy: "+str(accuracy))
    print("The predicted categories for X_test are:")
    print(predicted_cats[:5])
    print("The actual categories are:")
    print(y_test[:5])
    # print(predicted_cats == actual_cats) # shows a bool array of accuracy

# Start of data prediction
if True:
    knn_train = KNeighborsClassifier(n_neighbors=best_k)
    knn_train.fit(X_labeled, y_labeled)
    print("\nCreated and trained a knn classifier with k =", best_k)
    predicted_cats = knn_train.predict(X_unlabeled_full)
    predicted_cats2 = knn_train.predict(X_unlabeled_partial)
    print("The predicted categories for the unlabeled data are:")
    print(predicted_cats)
    print("The predicted categories for the unlabeled & missing data are:")
    print(predicted_cats2)

#
# Use iris4.py as your guide - it's "mostly" copy-and-paste
# HOWEVER -- there are points where things diverge...
# AND -- our goal is that you understand and feel more and more comfortable
#        with each of the parts of "the machine learning pipeline" ... !
# Also: for the digits data...
#     + the first 10 rows [0:10] are unlabeled AND have only partial data!
#     + the next 12 rows [10:22] are unlabeled but have full data... .
# You should create TWO sets of unlabeled data to be predicted.
#     + extra credit:  predict the missing pixels themselves!
#













"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?
  + how smoothly were you able to adapt from the iris dataset to here?
  + how high were you able to get the average cross-validation (testing) score?



Then, include the predicted labels of the 12 full-data digits with no label
Past those labels (just labels) here:

You'll have 12 digit labels:
['digit 9' 'digit 9' 'digit 5' 'digit 5' 'digit 6' 'digit 5' 'digit 0'
 'digit 3' 'digit 8' 'digit 9' 'digit 8' 'digit 4']


And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

Only use the top half to train your code. Do not train it using the bottom half.
This one had a lower testing-data score but both of them often had a 1.0 training-data score.

Past those labels (just labels) here:
You'll have 10 lines:






If you predicted the pixels themselves, cool! Share those, as well. (This is Ex. Cr.)


"""









#
# feature display - use %matplotlib to make this work smoothly
#


def show_digit( Pixels ):
    """ input Pixels should be an np.array of 64 integers (valued between 0 to 15) 
        there's no return value, but this should show an image of that 
        digit in an 8x8 pixel square
    """
    from matplotlib import pyplot as plt
    print(Pixels.shape)
    Patch = Pixels.reshape((8,8))
    plt.figure(1, figsize=(4,4))
    plt.imshow(Patch, cmap=plt.cm.gray_r, interpolation='nearest')  # plt.cm.gray_r   # plt.cm.hot
    plt.show()
    
# try it!
if False:
    row = 13
    Pixels = X_all[row][0:64]
    show_digit(Pixels)
    print("That image has the label:", y_all[row])

# another try
if False:
    row = 12
    Pixels = X_all[row][0:64]
    show_digit(Pixels)
    print("That image has the label:", y_all[row])


