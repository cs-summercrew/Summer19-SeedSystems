Logistic Regression with StatsModels

In this folder, you will find an example of a logistic regression on a csv file
using the python module StatsModels.

Dependencies:
To run this file, you must import the pandas library, the DataFrame library
from pandas, and the StatsModels API (statsmodels.formula.api).
Furthermore, in the command line, type the following commands:

pip install msgpack

pip install statsmodels

For futher clarification and for any issues, consult the following link to 
the StatsModels tutorial for installation:

https://www.statsmodels.org/stable/install.html

Overview:

The example provided consists of two parts. The first is a csv file called
Happiness.csv. This is a ficticious data set designed to represent the factors
that attribute to the happiness of an individual Harvey Mudd student. The first
column of the csv is whether a given student is happy or not, described
by either a 1 (happy) or a 0 (unhappy). The second represents the number of total puppies that
student has at the given time of their happiness state. The third represents
the total pounds of candy the student has. And the final represents the number
of rain clouds in the sky.

The python script attached takes this csv file, loads it into a pandas object, 
creates a DataFrame object from that, and finally fits it to a logistic
regression model using StatsModels commands. Running the script will print out
a summary page in the terminal detailing the coefficients of your logistic 
regression equation. For more information on decifering the summary page, 
consult these two websites:

https://mashimo.wordpress.com/2017/07/26/logistic-regression-with-python-statsmodels/

https://github.com/Mashimo/datascience/blob/master/01-Regression/LogisticRegressionSM.ipynb

Further information for StatsModels available at:

https://www.statsmodels.org/stable/index.html