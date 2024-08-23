# Import dependencies
from config import config

# Import libraries
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


#Load data from data/raw file
raw_data = pd.read_csv(config.DATA_PATH / 'raw/lifestyle_data.csv')

# Check columns for missing data
na_columns = []
for column in raw_data.columns:
    if raw_data[column].isna().sum() > 0:
        na_columns.append(column)

# Drop columns as required
raw_data.drop(columns = 'Timestamp',axis = 1, inplace=True) # Doesn't have direct relation with the 
raw_data.drop(columns=['LIVE_VISION', 'FLOW'], axis=1, inplace=True) # Column details unavailable

# Rename columns to correct format
raw_data.rename(columns={'DAILY_SHOUTING': 'AVERAGE_SHOUTING',
                         'DAILY_STRESS': 'AVERAGE_STRESS',
                         'DAILY_STEPS': 'AVERAGE_DAILY_STEPS'},
                inplace=True)

# Group columns into numerical and categorical
raw_data['BMI_RANGE'] = raw_data.BMI_RANGE.astype('O')
raw_data['SUFFICIENT_INCOME'] = raw_data.SUFFICIENT_INCOME.astype('O')
categorical_columns = [column for column in raw_data.columns if raw_data[column].dtype == 'O']
numerical_columns = [column for column in raw_data.columns\
                    if column not in categorical_columns and\
                    column != 'WORK_LIFE_BALANCE_SCORE']
raw_data.replace('1/1/00', 'Missing', inplace=True) # Clearing mislabeled data


# Save processed data for training
data = raw_data.copy()
data.to_csv('data/processed/lifestyle.csv')


