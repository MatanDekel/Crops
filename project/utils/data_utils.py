import pandas as pd


def load_data(file_path):
    df = pd.read_excel(file_path)
    return df


def preprocess_data(df):
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('Unnamed: 0.1', axis=1)

    # Filter the data for the "Center" region and "Spring" season
    df = df[(df['region'] == 'Center') & (df['season'] == 'Spring')]

    # Convert 'date' to datetime and extract 'month' and 'year'
    df['date'] = pd.to_datetime(df['date'], format='%m-%Y')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    return df
