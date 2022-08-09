# Zillow Clustering Project

## Project Goals

The goal is to find the key drivers of log error for single-family properties on Zillow, then make recommendations to help the data science team have a better prediction on the future log error.

## Project Description

In this report, we will analyze the Zillow 2017 single-family property transaction data, incorporate clustering methodologies and use the regression machine learning method to develop a model to prediction of the log error base on the selected features. Then give out the recommendations about how to improve the predictions for the log error and my next steps.

## Initial Questions

1. Does the log error related by location?
2. Why orange county has the biggest log error? is it because of the house price in orange county is different than average?
3. If the Orange county average price could impact log error, what about the relationship overall house value and log error?
4. Is there any relationship between quality type, square feet, room count, house age and log error?

## Data Dictionary

Variables are used in this analysis:

bedroom : Number of bedrooms in home.
bathroom : Number of bathrooms in home including fractional bathrooms.
parcelid : Unique identifier for parcels (lots) 
fips : Federal Information Processing Standard code.
yearbuilt: The Year the principal residence was built.
finished_square_ft : Calculated total finished living area of the home.
lot_square_ft : Area of the lot in square feet.
house_value : The total tax assessed value of the parcel.
structure_value : The assessed value of the built structure on the parcel.
land_value : The assessed value of the land area of the parcel
tax : The total property tax assessed for that assessment year.
garage : Total number of garages on the lot including an attached garage.
pool : Number of pools on the lot (if any).
quality_type : Overall assessment of condition of the building from best (lowest) to worst (highest).
latitude : Latitude of the middle of the parcel multiplied by 10e6.
longitude : Longitude of the middle of the parcel multiplied by 10e6.
city : City in which the property is located (if any).
log_error : log error=log(Zestimate)−log(SalePrice).
transaction_date : House transaction date.

## Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the Zillow table. Store that env file locally in the repository.
2. Clone my repo (including the acquire_zillow.py and prepare_zillow.py) (confirm .gitignore is hiding your env.py file)
3. Libraries used are pandas, matplotlib, seaborn, numpy, sklearn, scipy.stats, math, cluster and feature_selection.
4. You should be able to run zillow_clustering_report.

## Plan

## Wrangle

### Modules (acquire.py + prepare.py)

1. Write SQL query to acquire the data from mysql server, then test acquire function
2. Add to acquire.py module
3. Write code to clean data in notebook
4. Write code to split data in notebook
6. Merge functions in a single function & test
8. Add all 3 functions (or more) to prepare.py file
9. Import into notebook and test functions
10. There is also an impute function for train, validate and test set.

### Acquire the Zillow data

1. select basementsqft, bathroomcnt, bedroomcnt, buildingqualitytypeid, calculatedfinishedsquarefeet, fips, latitude, longitude, lotsizesquarefeet, regionidcity, yearbuilt, structuretaxvaluedollarcnt, taxvaluedollarcnt, landtaxvaluedollarcnt, taxamount, logerror, transactiondate, poolcnt, garagecarcnt from zillow.properties_2017 and rename the columns at the same time.
2. left join zillow.predictions_2017 and select transaction date < '2018'
3. then choose prepertylandusetypeid = 261 which is single family properties code

### Duplicates
drop duplicate transactions, only keep the last transaction

### Missing Values (report.ipynb)

Missing value: drop columns missing 60% and rows missing 75%.

### Convert data type
convert fips into int.

### Handle outliers
handle outliers with a function for 'finished_square_ft', 'lot_square_ft', 'structure_value', 'house_value', 'land_value','tax'

### Set up a cut-off line to analyze majority of data
bedroom <= 6, bathroom <= 6, house_value < 2,000,000
drop bedroom == 0 and bathroom == 0

### rename values and columns
rename fips value with county name and rename the fips column

### convert latitude and longitude

### Create new columns
age = (2017 - yearbuilt)
room_count = (df.bathroom + df.bedroom)

### Data Split (prepare.py (def function), report.ipynb (run function))

* train = 56%
* validate = 24%
* test = 20%

### Using your modules (report.ipynb)
once acquire.py and prepare.py are created and tested, import into final report notebook to be ready for use.

## Set the Data Context

In this report, there are 52,441 observations in 2017 zillow data and I will use 19 features for my analysis to predict the log error.

1. plot a log error distribution, x is log error, y is density
2. plot a scatter plot for location, x is latitude, y is longitude

## Explore

### Impute the data
* Before we go exploring our data, we need to impute the train set first, impute function is in the prepare_zillow.py
* This are how I impute my data:
    1. fill the quality type null with mean.
    2. drop the rest of null value.

1. Does the log error related by location?
2. Why orange county has the biggest log error? is it because of the house price in orange county is different than average?
3. If the Orange county average price could impact log error, what about the relationship overall house value and log error?
4. Is there any relationship between quality type, square feet, room count, house age and log error?

## Exploring through visualizations (report.ipynb)

1. Does the log error related by location?

* what is the log error mean for different county and city? plot a boxplot to show the difference between different county and city. X is county name and city, y is log error.
* run a INOVA statistic test to test the log error are different in those three counties.
* run a pearsonr statistic test to test the correlation for city and log error.

2. Why orange county has the biggest log error? is it because of the house price in orange county is different than average?

* what is the house value mean for each county? plot a boxplot to show the difference between different county. X is county name, y is house value.
* run a one sample  T-test to test to test Orange county average house price is higher than overall average house price.

3. If the Orange county average price could impact log error, what about the relationship overall house value and log error?

* what is the relationship between log error and house value? plot a scatter plot to show the relationship. X is house value, y is log error.

4. Is there any relationship between quality type, square feet, room count, house age and log error?

* scatter plot for age and log error, x is age, y is log error.
* scatter plot for finished square feet and log error, x is finished square feet, y is log error.
* bar plot for room count and log error, x is room count, y is log error.
* scatter plot for quality type and log error, x is log error, y is quality type.

## Summary (report.ipynb)

* Location and price are definitely related to log error. Quality type, square feet, room count and house age didn’t meet our confidence level with the statistical tests.
* The clusters I will create are:
    1. location: latitude, longitude, city, county.
    2. house size and age: finished_square_ft, lot_square_ft, room_count, age.
    3. price: structure_value, house_value, land_value.

# Clustering

### preparation for clustering
* split the data 
* get dummy variables for county
* scale the data

### Clusters:
    1. Cluster 1 : location (latitude, longitude, city, LA, Orange, Ventura)
    2. Cluster 2 : house size and age (finished_square_ft, lot_square_ft, room_count, age)
    3. Cluster 3 : price (structure_value, house_value, land_value)

* each clusters has a chart for select the best value of k and cluster grouping chart.

# Modeling

## Select Evaluation Metric (Report.ipynb)

Because house value is a continuous variable, I will use five different Regression machine learning models to fit the train. Those five models will use same features but different algorithms. Then evaluate on validate set for overfit and pick the best model on the test set.

The metrics I will use are RMSE (Root Mean Squared Error). RMSE is the most commonly used metric for regression model also it has the same unit as our target value.

* Features will be used are:

    'finished_square_ft', 'latitude', 'city', 'structure_value', 'room_count'.

## Evaluate Baseline (Report.ipynb)

The baseline value I set for train and validate set is the mean of log error on the train set.

## Develop 5 Models (Report.ipynb)

1. Linear Regression
2. Lasso-Lars
3. TweedieRegressor
4. 2nd degree Polynomial
5. interaction only polynomial

## Evaluate on Train (Report.ipynb)

caculate RMSE scores for each model, three top models are:

1. 2nd degree polynomial 
2. interaction only polynomial
3. linear Regression

## Evaluate on Validate (Report.ipynb)

None of the model seems to overfit.
caculate RMSE scores for each model, the best model is interaction only polynomial.

## Evaluate Top Model on Test (Report.ipynb)

test result:
* RMSE: 0.1721

## Expectation:
According to the test result, I expect there will be a 0.17 log error for my future prediction if the data source has no major change.

# Report (Final Notebook)

## Code commenting (Report.ipynb)

## Written Conclusion Summary (Report.ipynb)

By analyzing the key drivers of the majority of Zillow single house value of 2017, we create three clusters to explore and built an interaction only polynomial regression model with the top five attributes ('finished_square_ft', 'latitude', 'city', 'structure_value', 'room_count') from RFE method to predict the log error. The RMSE for the test set is 0.17. The best model on the test set doesn't beat the baseline of 0.16.

## Recommendations (Report.ipynb)

1. This project is very similar to the regression project, the only difference is we added clusters exploration. But from the RFE result, non of the clusters are useful. Therefore, I will say that maybe clustering is not the best approach for the Zillow data.
2. Also, the Linear Regression model performance is very poor. We can prove that the features we explore are related to log error, but the test result looks not good. It may be because the linear regression models are not the best fit for non-linear data like Zillow. We need to use different type of models to predict the log error.
3. There are a lot of missing data for the pool and garage, so it will be better to collect as much data as possible.

## conclusion next steps (Report.ipynb)

1. Even though clustering seems useless for this data, I still want to explore more about different clusters I can build to see if it's really not helping at all.
2. Since we already know that linear regression models don't perform well on Zillow data, I would like to build different types of modeld to have a better prediction of the target variable.

## no errors (Report.ipynb)

# Live Presentation

## Intro (live)

## audience & setting (live)

Data science team.

## content (live)

## Verbal Conclusion (findings, next steps, recommendations) (live)

## time (live)
5 minutes.