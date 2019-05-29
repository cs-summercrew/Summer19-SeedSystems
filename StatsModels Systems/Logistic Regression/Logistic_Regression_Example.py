# Necessary dependencies
import pandas as pd
from pandas import DataFrame
import statsmodels.formula.api as sm

# Create a pandas object using read_csv function, change PATH to fit where csv 
# file exists
PATH = 'Happiness_LogReg.csv'
happiness_data = pd.read_csv(PATH)

# Create a dataframe object, label the columns based on the csv columns
df = DataFrame(happiness_data,columns=['Happy', 'Num_Puppies', 
'Pounds_Candy', 'Num_RainClouds'])

# Set all of the dependent columns as 'X', set the independent column as 'Y'
X = df[['Num_Puppies','Pounds_Candy','Num_RainClouds']]
Y = df['Happy']

# Manually set the X intercept as 1, no need to use sm.add_constant()
X['intercept'] = 1.0

# Fit our model using the Logit() function from StatsModels
model = sm.Logit(Y,X)
result = model.fit()

# Print the results
print_model = result.summary()
print(print_model)
