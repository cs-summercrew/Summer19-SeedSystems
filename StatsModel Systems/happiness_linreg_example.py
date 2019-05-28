import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm

# Create a pandas object using read_csv function, change PATH to fit where csv 
# file exists
PATH = 'Happiness.csv'
happiness_data = pd.read_csv(PATH)

# Create a dataframe object, label the columns based on the csv columns
df = DataFrame(happiness_data,columns=['Happiness', 'Num_Puppies', 
'Pounds_Candy', 'Num_RainClouds'])

# Set all of the dependent columns as 'X', set the independent column as 'Y'
X = df[['Num_Puppies','Pounds_Candy','Num_RainClouds']]
Y = df['Happiness']

# Add a constant to our dependent columns
X = sm.add_constant(X)

# Fit our model using Ordinary Least Squares (OLS) method
model = sm.OLS(Y,X).fit()
predictions = model.predict(X)

# Obtain the summary of our data and print it
print_model = model.summary()
print(print_model)
