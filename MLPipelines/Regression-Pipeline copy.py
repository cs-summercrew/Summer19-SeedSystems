#
# digits5: modeling the digits data with DTs and RFs
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
# The "answers" to the 20 unknown digits, labeled -1:
#
answers = [9,9,5,5,6,5,0,9,8,9,8,4,0,1,2,3,4,5,6,7]


print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('digits5.csv', header=0)    # read the file w/header row #0
# df.head()                                 # first five lines
# df.info()                                 # column details
print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_all = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_all = df[ '64' ].values      # individually addressable columns (by name)
# now, model from iris5.py to try DTs and RFs on the digits dataset!
X_unlabeled = X_all[:20]
y_unlabeled = y_all[:20]
X_labeled = X_all[20:]
y_labeled = y_all[20:]
# we can scramble the data - but only the labeled data!
indices = np.random.permutation(len(X_labeled))
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

