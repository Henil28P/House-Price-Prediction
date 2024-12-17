# This script is for training the House Price Prediction model

# To install all the ML libraries, packages and its dependencies:
# Run the command on the terminal (if Python script): python3 -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn
# Make sure to be in the Python virtual environment and then run that command
# Steps for creating a Python virtual environment and activating it:
# 1. (On Mac): $ python3 -m venv .venv (to create the virtual environment)
# 2. (On Mac): $ source .venv/bin/activate (to go in the virtual environment and start installing packages and modules)

import pandas as pd
import numpy as np
import matplotlib as plt
# import sklearn as skl
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
housing.head() # output the first 5 rows of the data as a Pandas DataFrame

# Each row represents 1 district.
# There are 10 attributes (longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
# population, households, median_income, median_house_value, and ocean_proximity)

# Note: If you get the error "SSLCertVerificationError: certificate verify failed",
# This is because Python cannot verify the SSL certificate of the URL you're trying to access.
# This issue is often due to missing or outdated SSL certificates on your system or specific configurations in your Python environment

# Solution for Mac: Run the command $ /Applications/Python\ 3.x/Install\ Certificates.command on the terminal
# For this command, replace 'x' with the version of the Python you are using currently such as 3.11
# The above command installs/updates your Python SSL certificate in your Mac machine