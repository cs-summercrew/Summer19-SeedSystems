#
# your own data modeling... 
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
#  Let us know which data you're using 
#    + and which columns you're considering features/labels!
#

#
# This is taken from the titanic example...
#
# Be sure to adapt to your own data!
#
print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('titanic5.csv', header=0)    # read the file w/header row #0
#
# drop columns here
#
df = df.drop('name', axis=1)  # axis = 1 means column

df.head()                                 # first five lines
df.info()                                 # column details

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
#
#

print("+++ Start of numpy/scikit-learn +++\n")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays 
X_all = df.drop('survived', axis=1).values       
y_all = df[ 'survived' ].values      


