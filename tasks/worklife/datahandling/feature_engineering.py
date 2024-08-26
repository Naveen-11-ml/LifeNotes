# Import dependencies
from config import config
from tasks.worklife.datahandling import preprocessing as preprocessed_data
# Import Required Library
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# Load data into a pandas DataFrame
data = pd.read_csv(config.DATA_PATH / 'processed/lifestyle.csv')

# Split training and test data set
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('WORK_LIFE_BALANCE_SCORE', axis=1),
    data['WORK_LIFE_BALANCE_SCORE'],
    test_size=0.15, random_state=0)

# Load required data from previous process
categorical_columns = preprocessed_data.categorical_columns
numerical_columns = preprocessed_data.numerical_columns


# Mark values 'Rare' within the data with very low frequency to avoid rare cases
def value_frequency(df, var, perc):
    df = df.copy()
    temp = df.groupby(var)[var].count() / len(df)
    return temp[temp > perc].index

for column in categorical_columns:
    rare_items = value_frequency(X_train, column, 0.05)
    X_train[column] = X_train[column].where(X_train[column].isin(rare_items), 'Rare')
    X_test[column] = X_test[column].where(X_test[column].isin(rare_items), 'Rare')

# Separate different types of columns with respect to transformation
ordinal_encoding_columns = categorical_columns[:3] # These features required cardinality
one_hot_encoding_columns = categorical_columns[3:] # String categorical features
minmax_scaler_columns = numerical_columns # discrete features

column_transformer = ColumnTransformer(transformers=[
    ('label_encoder', OrdinalEncoder(), ordinal_encoding_columns),
    ('one_hot_encoder', OneHotEncoder(), one_hot_encoding_columns),
    ('minmaxscalar', MinMaxScaler(), minmax_scaler_columns)
    ], remainder='passthrough')

target_scalar = MinMaxScaler()
y_train = target_scalar.fit_transform(pd.DataFrame(y_train))

# Save data to train the model and transformers
joblib.dump(target_scalar, config.TASK_PATH / 'worklife/datahandling/targer_scalar.joblib')
joblib.dump(column_transformer, config.TASK_PATH / 'worklife/datahandling/column_transformer.joblib')
joblib.dump(X_train, config.TASK_PATH / 'worklife/data/X_train.joblib')
joblib.dump(X_test, config.TASK_PATH / 'worklife/data/X_test.joblib')
joblib.dump(y_train, config.TASK_PATH / 'worklife/data/y_train.joblib')
joblib.dump(y_test, config.TASK_PATH / 'worklife/data/y_test.joblib')