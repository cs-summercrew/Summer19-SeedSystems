# Authors: CS-World Domination Summer19 - DM
# Library Imports
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import utils, metrics, datasets, preprocessing
from sklearn import model_selection # MinMaxScaler,StandardScaler
from sklearn.feature_selection import f_regression,mutual_info_regression,SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# ML models Imports
from sklearn import svm, linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor

# NOTE: I strongly recommend that you skim the README and its additional resources before 
#       looking at my code and as a reference if you get confused at any point

#####################################################################################
########################## Start of main() functions ################################
#####################################################################################

def loadData():
    """Loads data from a csv and gets it into a workable format.
       The size param specifies how much of the data to split into testing/training"""
    
    # Load the data
    df = pd.read_csv('auto-complete.csv', header=0)
    # Create/modify new features and drop unused ones
    df = featurefy(df)
    df = df.drop('car name', axis=1)
    df = df.drop('model year', axis=1)
    df = onehot(df)
    # Visualize Data
    visualizeData(df)
    # Check for multicollinearity
    multicollCheck(df)
    # NOTE: You could drop the feature with the highest VIF, but given 
    # how few features we're working with I decided to leave it be

    # Check for the best features being used
    featureSelect(df, 7)
    # NOTE: Based on the results of the feature selection, I choose to drop the Japanese and European vars
    #       The thinking goes that both regions probably had better environmental regulations than the US,
    #       So all that really matters is whether the car was made in the US or not. 
    df = df.drop('european', axis=1)
    df = df.drop('japanese', axis=1)

    # Load the "unknown" data that we will check our models against
    # NOTE: There were missing mpg values in the original data, found some/made best guess online using fuelly.com
    df2 = pd.read_csv('auto-missing.csv', header=0)
    df2 = featurefy(df2)
    df2 = onehot(df2)
    df2 = df2.drop('model year', axis=1)
    df2 = df2.drop('car name', axis=1)
    df2 = df2.drop('european', axis=1)
    df2 = df2.drop('japanese', axis=1)
    X_unknown = df2.iloc[:,1:].values
    y_unknown = df2[ 'mpg' ].values
    
    # Organizing data into training/testing
    # NOTE: .values converts df to numpy array
    X_data = df.iloc[:,1:].values           # iloc == "integer locations" of rows/cols
    y_data = df['mpg'].values               # individually addressable columns (by name)

    return X_data, y_data, X_unknown, y_unknown

def scaleData(X_data):
    """Scales data in two different ways"""

    # MinMaxScaler subtracts the feature's mean from each value and then divides by the range.
    """Normalization is useful when your data has varying scales and the algorithm 
    you are using does not make assumptions about the distribution of your data, 
    such as k-nearest neighbors and artificial neural networks."""
    # mm_scaler = preprocessing.MinMaxScaler()
    # X_data[:4] = mm_scaler.fit_transform(X_data[:4])
    
    # StandardScaler scales each feature to have mean of 0 and standard deviation of 1.
    """Standardization is useful when your data has varying scales and the algorithm 
    you are using does make assumptions about your data having a Gaussian distribution, 
    such as linear regression, logistic regression and linear discriminant analysis"""
    # s_scaler = preprocessing.StandardScaler()
    # X_data[:,:4] = s_scaler.fit_transform(X_data[:,:4])

    rb_scaler = preprocessing.RobustScaler()
    X_data[:4] = rb_scaler.fit_transform(X_data[:4])

    # norm = preprocessing.Normalizer()
    # X_data[:4] = norm.fit_transform(X_data[:4])

    return X_data

def crossValidation(X_train, y_train):
    """Does cross validation tests on the data to help determine the best model"""

    print("\n\n+++ Comparing algorithms with cross-validation! +++")
    # Make a list models to cross-validate
    models = []
    models.append( ("Decision Trees     ",DecisionTreeRegressor()) )
    models.append( ("Random Forests     ",RandomForestRegressor(n_estimators=20)) )
    models.append( ("Very Random Forests",ExtraTreesRegressor(n_estimators=20)) )
    models.append( ("OLS                ",linear_model.LinearRegression()) )
    models.append( ("SVR                ",svm.SVR(gamma="scale")) )
    models.append( ("BayesianRidge      ",linear_model.BayesianRidge()) )
    models.append( ("PassiveAggressive  ",linear_model.PassiveAggressiveRegressor()) )
    models.append( ("SGD                ",linear_model.SGDRegressor()) )

    # Loop through and evaluate each model
    r2Results = []
    maeResults = []
    rmseResults = []
    names = []
    rankList1 = []
    rankList2 = []
    scoring = ['r2','neg_mean_absolute_error', 'neg_mean_squared_error']
    print("Note that results show the means (of the cross-validating) for each scoring metric, with standard deviation in parenthesis")
    print("\nAlgorithm          : R"+chr(0x00B2)+", Mean Absolute Error, Root Mean Squared Error")
    for name, model in models:
        # kfold validation
        kfold = model_selection.KFold(n_splits=5, random_state=None)
        cv_scores = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        # Record the results of each metric for printing/visualization
        # NOTE: cross_validate returns dictionaries (cv_scores is a dictionary)
        r2 = cv_scores["test_r2"]
        mae = (-1)*cv_scores["test_neg_mean_absolute_error"]
        mse = (-1)*cv_scores["test_neg_mean_squared_error"]
        rmse = np.array(list(map(lambda x: math.sqrt(x), mse)))
        r2Results.append(r2)
        maeResults.append(mae)
        rmseResults.append(rmse)
        names.append(name.strip())
        rankList1.append([name.strip(), r2.mean()])
        rankList2.append([name.strip(), mae.mean()])
        print( "%s: %0.3f (%0.3f)," % (name, r2.mean(), r2.std()),
               "%.3f (%0.3f)," % (mae.mean(), mae.std()),
               "%.3f (%0.3f)" % (rmse.mean(), rmse.std()) )
    # Summarization/Analysis of results
    rankList1 = sorted( rankList1, key=lambda x: x[1], reverse=True )
    rankList2 = sorted( rankList2, key=lambda x: x[1], reverse=False )
    print("The best algorithm after ranking R"+chr(0x00B2)+" is: "+rankList1[0][0])
    print("The best algorithm after ranking MAE is: "+rankList2[0][0])
    if False:
        # The box and whisker plots help show the spread of values from cross-validation
        boxPlot(r2Results, names, "R Squared")             # Larger (higher) is better
        boxPlot(maeResults, names, "Mean Absolute Error")  # Smaller (lower) is better
        boxPlot(rmseResults, names, "Mean Squared Error")  # Smaller (lower) is better
    return

def trainModel(X_train, y_train, X_test, y_test):
    """This function fine-tune your chosen algorithm (the best alg is probably very random forests or OLS)
       Fine-tuning here is faster/less intensive than in the cross-validation function, but it could also be
       done there as well."""
    print("\n\n+++ Predicting testing data! +++")
    
    # Choose model
    # model = ExtraTreesRegressor(n_estimators=100)
    model = linear_model.LinearRegression()
    
    # Plot Residuals (Errors)
    if False:
        """A common use of the residuals plot is to analyze the variance of the error of the regressor. 
        If the points are randomly dispersed around the horizontal axis, a linear regression model is usually 
        appropriate for the data; otherwise, a non-linear model is more appropriate."""
        from yellowbrick.regressor import ResidualsPlot
        visualizer = ResidualsPlot(model, hist=False)
        visualizer.fit(X_train, y_train)  # Fit the training data to the model
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.poof()                 # Draw/show/poof the data
    
    # Check model results on test data
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Note that Regression tends to give imprecise predictions, so expect to never get perfect answers.\n" +
    "Printing only the first 10 values for readability. The printed MAE is calculated from all the data.\n")
    print( "Prediction         :", list(map(lambda x: float("%.1f"%x),predictions[:10])) )
    print( "Actual             :", list(map(lambda x: float("%.3f"%x), y_test[:10])) )
    ErrorList = []
    for i in range(len(predictions)):
        ErrorList.append( abs(predictions[i]-y_test[i]) )
    # Check histogram distribution of errors
    # pd.Series(ErrorList).hist()
    # plt.show()
    print( "Absolute Errors    :", list(map(lambda x: float("%.1f"%x), ErrorList[:10])) )
    print( "Mean Absolute Error:", round(metrics.mean_absolute_error(y_test,predictions), 1) )
    print( "Absolute Error Std :", round(np.array(ErrorList).std(), 1) )
    return

def predictUnknown(X_data, y_data, X_unknown, y_unknown):
    """Makes predictions on the unknown data"""
    print("\n\n+++ Predicting unknown data! +++")
    # model = ExtraTreesRegressor(n_estimators=100)
    model = linear_model.LinearRegression()
    model.fit(X_data, y_data)
    predictions = model.predict(X_unknown)
    print("Note that since the actual values are mostly a best-guess estimation of mine, "+
    "it makes sense that the errors will be 'larger' than on the testing data.\n")
    print("Prediction         :",list(map(lambda x: float("%.1f"%x),predictions)))
    print("Actual             :",list(map(lambda x: float("%.3f"%x), y_unknown)))
    ErrorList = []
    for i in range(len(predictions)):
        ErrorList.append(predictions[i]-y_unknown[i])
    ErrorList[-1] = "Na"
    print("Absolute Errors    :",list(map(lambda x: x if type(x)==str else abs(float("%.1f"%x)), ErrorList)))
    print("Mean Absolute Error: ",round(metrics.mean_absolute_error(y_unknown[:-1],predictions[:-1]), 1))
    return

#####################################################################################
########################### End of main() functions #################################
#####################################################################################

#####################################################################################
########################## Start of Helper functions ################################
#####################################################################################

def visualizeData(df):
    """It is often a good idea to visualize data before starting to working with it."""
    print("\n+++ Visualizing the feature data! +++")
    # The scatterplot is unreadable with all the categorical data in it, so I changed weight to a float here
    # because its the only non-categorical category that wasnt a float
    df = df.astype({"weight": float})
    numeric_df = df.select_dtypes(include=['float64'], ).copy() # Lets you select specific types of data from your df
    cat_df = df.select_dtypes(exclude=['float64'], ).copy()
    if False:
        cat_df.hist()
        numeric_df.hist()
    if False:
        from pandas.plotting import scatter_matrix
        scatter_matrix(numeric_df)
    if False:
        numeric_df.plot(kind='density', subplots=True, layout=(2,2), sharex=False)
    plt.show()
    return

def featurefy(df):
    """ Modifies the missing values in the horsepower feature. Also creates two new features, diesel and station wagon,
        from car name"""
    # NOTE: Six values are missing horsepower, so we replace those values with the column's mean
    #       Alternatively we could just drop the entries with missing data using: df = df.dropna()
    df['horsepower'] = df['horsepower'].replace('?', np.NaN)
    df['horsepower'] = df['horsepower'].map(np.float64)
    df['horsepower'].fillna( df['horsepower'].mean(), inplace=True )
    # Create new features
    dieselList = []
    swList = []
    for name in df['car name']:
        dieselList.append(1) if "diesel" in name else dieselList.append(0)
        swList.append(1) if ("(sw)" in name) or ("wagon" in name) else swList.append(0)
    df['diesel'] = pd.Series(dieselList)
    df['station wagon'] = pd.Series(swList)
    return df

def onehot(df):
    """One hot encodes the features: cylinders and origin"""
    # NOTE: I would recommend doing the encoding instead with pd.get_dummies, 
    #       but since I call this function twice, there are some bugs if I use it here.
    # One-hot encode cylinders
    cyl4 = []
    cyl6 = []
    cyl8 = []
    for num in df['cylinders']:
        cyl8.append(1) if 8 == num else cyl8.append(0)
        cyl6.append(1) if 6 == num else cyl6.append(0)
        cyl4.append(1) if 4 == num else cyl4.append(0)
    df['4 cylinders'] = pd.Series(cyl4)
    df['6 cylinders'] = pd.Series(cyl6)
    df['8 cylinders'] = pd.Series(cyl8)
    df = df.drop('cylinders', axis=1)
    # One-hot encode origin
    amer = []
    euro = []
    jap = []
    for num in df['origin']:
        amer.append(1) if 1 == num else amer.append(0)
        euro.append(1) if 2 == num else euro.append(0)
        jap.append(1) if 3 == num else jap.append(0) 
    df['american'] = pd.Series(amer)
    df['european'] = pd.Series(euro)
    df['japanese'] = pd.Series(jap)
    df = df.drop('origin', axis=1)
    return df

def multicollCheck(df):
    """ Checks for multicollinearity using VIF scores, the included link explains when checking is important"""
    print("\n+++ Checking for multicollinearity! +++")
    print("Note that I was unable to get the warning to go away, so that task is left to the reader!")
    # Check for multicollinearity!
    # A rule of thumb is that if there are VIF scores of more than five to ten, the variables are multicollinear!!!
    # However, do know that (rarely) there can be low VIF's while still have multicollinearity...
    df = df.drop('mpg', axis=1)
    # Drop the one-hot variables so they aren't checked: dummy variables will always have high VIF's
    df = df.drop('4 cylinders', axis=1)
    df = df.drop('6 cylinders', axis=1)
    df = df.drop('8 cylinders', axis=1)
    df = df.drop('american', axis=1)
    df = df.drop('european', axis=1)
    df = df.drop('japanese', axis=1)
    # Add a regression Intercept
    # Based off of info from this post: https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    X = add_constant(df)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],index=X.columns)
    print(vif)
    return

def featureSelect(df, numToSelect):
    """ Univariate feature selection using scikit's SelectKBest and f_regression"""
    print("\n+++ Selecting the",numToSelect,"best features! +++")
    # Create and fit selector
    selector = SelectKBest(f_regression, k=numToSelect)
    features = df.iloc[:,0:-1]
    selector.fit(features, df["mpg"])
    # Get columns to keep
    cols = selector.get_support(indices=True)
    features = features.iloc[:,cols]
    print("The",numToSelect,"best features are:",features.columns.values)
    return

def boxPlot(results, names, metric):
    """ This box plot shows the spread of the data, NOT the confidence interval!!! 
        The box extends from the lower to upper quartile values of the data, with a line at the median. 
        The whiskers extend from the box to show the range of the data. Dots are outliers"""
    fig = plt.figure()
    fig.suptitle('Algorithm '+metric+' Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    return

#####################################################################################
########################### End of Helper functions #################################
#####################################################################################

def flaskPrediction(X_data, y_data, info):
    """Given an input attribute list, this function returns a prediction based on the chosen model """
    print("\n+++ Predicting the input data! +++")
    # model = ExtraTreesRegressor(n_estimators=100)
    model = linear_model.LinearRegression()
    info = np.array(info)
    info = info.reshape(1, -1)
    model.fit(X_data, y_data)
    prediction = model.predict(info)
    print("With this input, the predicted mpg is:", round(prediction[0], 1))
    print(prediction)
    return prediction[0]

def main():
    # Data Pre-processing
    (X_data, y_data, X_unknown, y_unknown) = loadData()     # Loads the csv file, input sets training size
    (X_data) = scaleData(X_data)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_data, y_data, 
    test_size=0.20, shuffle=True, random_state=None)        # Shuffle the data & split into test/train
    # Model Selection/Refinement
    crossValidation(X_train, y_train)                       # Compare different algorithms
    trainModel(X_train, y_train, X_test, y_test)            # Run/Refine the best algorithm on the test/train data
    # Test prediction on unknown data
    predictUnknown(X_data, y_data, X_unknown, y_unknown)
    info = [302,140.0,4294,16,0,0,0,0,1,1]
    flaskPrediction(X_data, y_data, info)

if __name__ == "__main__":
    main()