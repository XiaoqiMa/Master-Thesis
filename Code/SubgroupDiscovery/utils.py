from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype


def normalize_data(data):
    df = data.copy()
    num_features = [c for c in df.columns if is_numeric_dtype(df[c])]
    df[num_features] = preprocessing.scale(df[num_features])
    return df
