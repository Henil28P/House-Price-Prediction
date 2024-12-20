# This script is for training the House Price Prediction model

# To install all the ML libraries, packages and its dependencies:
# Run the command on the terminal (if Python script): python3 -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn
# Make sure to be in the Python virtual environment and then run that command
# Steps for creating a Python virtual environment and activating it:
# 1. (On Mac): $ python3 -m venv .venv (to create the virtual environment)
# 2. (On Mac): $ source .venv/bin/activate (to go in the virtual environment and start installing packages and modules)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # $ pip install scikit-learn
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
# import seaborn as sbn
import os
import tarfile
from six.moves import urllib

# If running this script as a Jupyter notebook by cell breakpoints, then you might need to run the following installation commands on a separate cell:
# $ !pip install <library> (to run in a shell as a command)
# OR
# $ %pip install <library> (provided by IPython kernel)
# There are different limitations and purposes for installing with ! and %
# For the purpose of this project, these libraries and dependencies have been installed separately on a terminal in an active virtual environment

# Initialise important files and paths
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(os.path.abspath(".."), "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Function to fetch the housing data - automating this process to install the dataset on multiple machines
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True) # target directory
    tgz_path = os.path.join(housing_path, "housing.tgz") # define download path
    urllib.request.urlretrieve(housing_url, tgz_path) #  download path
    housing_tgz = tarfile.open(tgz_path) # open the downloaded tar file
    housing_tgz.extractall(path=housing_path) # extract the tarball to housing_path
    housing_tgz.close() # close the tar file

# When calling the above function, it creates a datasets/housing directory in the project (outside Model Training directory),
# downloads the 'housing.tgz' file and extracts the 'housing.csv' file from it in this directory
fetch_housing_data()

# Function for loading the data using Pandas
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path) # returns a pandas DataFrame object containing all the data

housing = load_housing_data()

# Note: If you get the error "SSLCertVerificationError: certificate verify failed",
# This is because Python cannot verify the SSL certificate of the URL you're trying to access.
# This issue is often due to missing or outdated SSL certificates on your system or specific configurations in your Python environment

# Solution for Mac: Run the command $ /Applications/Python\ 3.x/Install\ Certificates.command on the terminal
# For this command, replace 'x' with the version of the Python you are using currently such as 3.11
# The above command installs/updates your Python SSL certificate in your Mac machine

# Each row represents 1 district.
# There are 10 attributes (longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
# population, households, median_income, median_house_value, and ocean_proximity)
def review_data(housing):
    housing.head() # output the first 5 rows of the data as a Pandas DataFrame

    housing.info() # get a quick description of the housing data

    # The field "ocean_proximity" is not numerical like all the others, its type is "Object"
    # Check how many districts belong to each category of the attribute "ocean_proximity"
    housing["ocean_proximity"].value_counts()

    # Use the describe() method to see the summary of the numerical attributes
    housing.describe()

    # Use %matplotlib inline to run in a separate cell in a Jupyter notebook environment (not in a script) as it tells Jupyter to set up Matplotlib using Jupyter's own backend
    # For this script and project, the command $ pip install matplotlib was used in a Python virtual environment
    housing.hist(bins=50, figsize=(20,15)) # hist() to plot a histogram for each numerical attribute
    plt.show()

review_data(housing)

############################## Part 2 - Create a test set ############################################

# This involves picking some instances randomly and set them aside (ie. 20% of dataset or less if the dataset is very large)

# --> $ pip install scikit-learn to install it on normal terminal while being in a virtual environment
# --> $ %pip install scikit-learn to install it in a Jupyter notebook for using sklearn dependencies in any cell

# Use train_test_split of sklearn module to split the data into 20% test and 80% train
def test_set(housing):
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # Check the amount of districts (data points) for each of the 2 sets
    len(train_set) # 80% of overall data = 16512
    len(test_set) # 20% of overall data = 4128

    # Suppose the median_income attribute (continuous numerical) is very important to predict median housing prices
    # Create an income category attribute with 5 categories labelled from 1 to 5
    housing["income_cat"] = pd.cut(housing["median_income"],
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1,2,3,4,5])

    # np.inf is NumPy's way to consider all income values above $60,000 to infinity

    # Plot histogram of income categories
    housing["income_cat"].hist()
    
    return housing

test_set(housing)

# Perform stratified sampling based on the income category using Scikit-Learn's StratifiedShuffleSplit class imported earlier
def stratified_sampling(housing):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        # Extract the rows corresponding to the training and test indices
        strat_train_set = housing.iloc[train_index]
        strat_test_set = housing.iloc[test_index]

    # Check the proportion of each income category in the stratified test set
    strat_test_set["income_cat"].value_counts() / len(strat_test_set)

    # Remove the income_cat attribute so the data is back to its original state
    for set_ in (strat_train_set, strat_test_set):
        # set_.drop("income_cat", axis=1, inplace=True)
        # set_ = housing[some_condition].copy()
        set_ = set_.drop("income_cat", axis=1)

    return strat_train_set, strat_test_set

strat_train_set, strat_test_set = stratified_sampling(housing)


################################## Part 3 - Discover and Visualise the Data to Gain Insights ###################################

# Firstly, put the test set aside and explore the training set
# Create a copy so that you can play with it without harming the training set
housing = strat_train_set.copy()

# 1. Visualising Geographical data
# Create a scatterplot of all districts to visualise the data since there is geographical info (latitude and longitude attributes)
def visualise_geodata(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) # alpha=0.1 makes it easier to visualise the places where there is a high density of data points

    # Make the patterns more stand out by adding more parameters and colourfully visualise 'median_house_value'
    # 's' (the radius of each circle) shows the district's population, 'c' (color) represents the price,
    # 'cmap' to use a predefined color map (ie. 'jet') and it ranges from blue (low values) to red (high prices)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                s=housing["population"]/100, label="population", figsize=(10,7),
                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    )
    plt.legend()

# Call the visualise_geodata() to see all relevant output
visualise_geodata(housing)

def observe_correlations(housing):
    # Select only numerical columns for correlation
    numerical_housing = housing.select_dtypes(include=[np.number])

    # Compute the 'standard correlation coefficient' (Pearsons's r) between every pair of attributes using the corr() method
    corr_matrix = numerical_housing.corr()

    # display the correlation matrix
    return corr_matrix["median_house_value"].sort_values(ascending=False)

# Call the observe_correlations() function to see all relevant output
observe_correlations(housing)

# Another way to check the correlation between attributes is to use pandas scatter_matrix() function
# It plots every numerical attribute against every other numerical attribute.

# 11 numerical attributes = 11^2=121 plots will not fit on a page
# So focus on only 4 main attributes that seem most correlated with the median housing value
def plot_scatter_matrix(housing):
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12,8))

    # From above scatter matrix, it looks like the most important attribute to predict the median house value is median income so zoom in on its correlation scatterplot
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# Call the scatter_matrix() function to see all relevant scatter_matrix
plot_scatter_matrix(housing)

# Before preparing the data for ML algorithms, try various attribute combinations

# eg. total rooms in a district is not useful if you don't know how many households there are, so you want total rooms per household
def attribute_combinations(housing):
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"] = housing["population"]/housing["households"]

    # look at correlation matrix again
    numerical_housing = housing.select_dtypes(include=[np.number])
    corr_matrix = numerical_housing.corr()
    return corr_matrix["median_house_value"].sort_values(ascending=False)

# Call the attribute_combinations() function to see all attributes after adding new ones
attribute_combinations(housing)


################################ Part 4 - Prepare the Data for Machine Learning Algorithms ###################################

# Revert to a clean training set
housing = strat_train_set.drop("median_house_value", axis=1)
# ==> Note: .drop() creates a copy of the data and does not affect 'strat_train_set'

# Separate the predictors and labels to not apply same transformationss to the predictors and the target values
housing_labels = strat_train_set["median_house_value"].copy()


def data_cleaning(housing):
    # Most ML algorithms don't work with missing features, in this, total_bedrooms attribute has some missing values

    # Option 1: get rid of the corresponding districts which have missing values: housing.dropna(subset=["total_bedrooms"])
    # Option 2: get rid of the entire attribute: housing.drop("total_bedrooms", axis=1)
    # Option 3: set the missing values to some value (0,mean,median) - impute it
    # median = housing["total_bedrooms"].median()
    # housing.fillna({"total_bedrooms": median}, inplace=True)

    # Apart from all the above manual options, use sklearn's SimpleImputer class from its 'impute' module
    imputer = SimpleImputer(strategy="median") # replace each attribute's missing value with its "median"

    # Create a copy of the data without the text attribute "ocean_proximity" since median can only be computer on numerical attributes
    housing_num = housing.drop("ocean_proximity", axis=1)
    housing_num = housing_num.drop("income_cat", axis=1)

    # Fit the 'imputer' instance to the training data using fit() method
    imputer.fit(housing_num)

    # Median is now stored in the 'statistics_' instance variable by 'imputer'
    imputer.statistics_
    housing_num.median().values

    # Use the trained imputer to transform the training set by replacing missing values with the learned median
    X = imputer.transform(housing_num)

    # Convert the above NumPy array result of transformed features to Pandas DataFrame
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    return housing_tr

data_cleaning(housing)

def handle_text_attributes(housing):

    housing_cat = housing[["ocean_proximity"]] # [[]] as OrdinalEncoder requires 2D input (a DataFrame or a 2-D array)
    housing_cat.head(10) # output first 10 rows of "ocean_proximity" categorical attribute

    # As most ML algorithms prefer to work with numbers,
    # 1. Ordinal Encoding: use sklearn's OrdinalEncoder class of its 'preprocessing' module to convert the above categorical values from text to numbers
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    housing_cat_encoded[:10]

    # get list of categories which is a list containing a 1D array of categories for each categoricaal attribute
    ordinal_encoder.categories_
    # return ordinal_encoder
    
    # 2. One-hot encoding: use sklearn's OneHotEncoder class of its 'preproccessing' module
    # Convert categorical values into one-hot vectors
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    housing_cat_1hot # output will be SciPy sparse matrix instead of a NumPy array
    # Sparse matrix only stores the location of the nonzero elements which saves tons of memory
    
    # Convert the above sparse matrix to a (dense) NumPy array with toarray() method
    housing_cat_1hot.toarray()
    
    # both will have same output of list of categories
    return ordinal_encoder.categories_, cat_encoder.categories_

handle_text_attributes(housing)