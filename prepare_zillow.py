import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['parcelid', 'num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def handle_missing_values(df, prop_required_columns=0.6, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df

def iqr_outliers(df, k=1.5, col_list=None):
    if col_list != None:
        for col in col_list:
            q1, q3 = df[col].quantile([.25, .75])
            iqr = q3 - q1
            upper_bound = q3 + k * iqr
            lower_bound = q1 - k * iqr
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    else:
        for col in list(df):
            q1, q3 = df[col].quantile([.25, .75])
            iqr = q3 - q1
            upper_bound = q3 + k * iqr
            lower_bound = q1 - k * iqr
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return df

def clean_zillow(df):
    df = df.sort_values('transaction_date').drop_duplicates('parcelid',keep='last')
    df = df[(df['bedroom'] > 0) & (df['bathroom'] > 0)]
    df = handle_missing_values(df)
    df.fips = df.fips.astype(int)
    df.fips = np.where(df.fips == 6037, 'LA', df.fips)
    df.fips = np.where(df.fips == '6059', 'Orange', df.fips)
    df.fips = np.where(df.fips == '6111', 'Ventura', df.fips)
    df = df.rename(columns=({'fips':'county'}))
    outlier_cols = ['finished_square_ft', 'lot_square_ft', 'structure_value', 'house_value', 'land_value','tax']
    df = iqr_outliers(df, col_list=outlier_cols)
    df = df[df.bedroom <= 6]
    df = df[df.bathroom <= 6]
    df = df[df.house_value < 2000000]
    df.latitude = df.latitude / 1_000_000
    df.longitude = df.longitude / 1_000_000
    df['age'] = 2017 - df.yearbuilt
    df['room_count'] = df.bathroom + df.bedroom
    return df

def split_data(df):
    train_validate, test = train_test_split(df,test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate,test_size=0.3, random_state=123)
    
    return train, validate, test

def impute_missing_value_zillow(df):
    df['quality_type'] = df.quality_type.fillna(round(df.quality_type.mean()))
    df.dropna(inplace=True)
    return df