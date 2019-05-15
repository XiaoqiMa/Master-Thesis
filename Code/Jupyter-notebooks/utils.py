from collections import defaultdict
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
class DataEncoding:

    def __init__(self, data):
        self.data = data
            
    def get_cat_features(self, data):
        cat_features = [c for c in data.columns if is_string_dtype(data[c])]
        return cat_features
    
    def get_num_features(self, data):
        num_features = [c for c in data.columns if is_numeric_dtype(data[c])]
        return num_features
    
    def label_encoding(self):
        d = defaultdict(LabelEncoder)
        cat_features = self.get_cat_features(self.data)
        df = self.data.copy()
        for name in cat_features:
            df[name] = df[[name]].apply(lambda x: d[name].fit_transform(x))
        return df
            
    def discretization(self, quantiles=10):
        num_features = self.get_num_features(self.data)
        df = self.data.copy()
        for name in num_features:
            df[name] = pd.qcut(df[name], quantiles , duplicates='drop').astype(str)
        return df
    
    def onehot_encoding(self):
        df = self.data.copy()
        return pd.get_dummies(df)