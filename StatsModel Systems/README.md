Linear Regression with StatsModels

In this folder, you will find an example of a linear regression on a csv file
using the python module StatsModels.

Dependencies:
To run this file, you must import the pandas library, the DataFrame library
from pandas, and the StatsModels API (statsmodels.api).
Furthermore, in the command line, type the following commands:

pip install msgpack
pip install statsmodels

For futher clarification and for any issues, consult the following link to 
the StatsModels tutorial for installation:
https://www.statsmodels.org/stable/install.html

Overview:

The example provided consists of two parts. The first is a csv file called
Happiness.csv. This is a ficticious data set designed to represent the factors
that attribute to the average happiness of a Harvey Mudd student. The first
column of the csv is the average weighted happiness of students at Mudd,
described on a 10 point scale with 10 being the happiest. The second 
represents the number of total puppies on campus at the given time of the
happiness score. The third represents the total pounds of candy on campus. And
the final represents the number of rain clouds in the sky.

The python script attached takes this csv file, loads it into a pandas object, 
creates a DataFrame object from that, and finally fits it to a linear
regression model using StatsModels commands. Running the script will print out
a summary page in the terminal detailing the coefficients of your linear 
regression equation. For more information on decifering the summary page, 
consult these two websites:
https://datatofish.com/statsmodels-linear-regression/
https://www.statsmodels.org/stable/regression.html

Further information for StatsModels available at:
https://www.statsmodels.org/stable/index.html