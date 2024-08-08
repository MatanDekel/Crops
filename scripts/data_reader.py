from utils.data_utils import load_data
import pandas as pd


def reader(path):
    df = pd.read_excel(path)
    return df
