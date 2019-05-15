import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pysubgroup as ps

def read_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    # simply drop missing values
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data


# def effect_change(data, features, attr, target, classifier):
#     train_data = pd.get_dummies(data[features], drop_first=True)
#     target_data = data[target]
#     df_attr = pd.get_dummies(data[attr], drop_first=True)
#     col_name = df_attr.columns[0]
#
#     flip_data = train_data.copy()
#     train_data = train_data.join(df_attr, how='inner')
#     flip_attr = df_attr.apply(lambda x: x ^ 1)
#
#     flip_train_data = flip_data.join(flip_attr, how='inner')
#     model = classifier.fit(train_data.values, target_data.values.ravel())
#     prob = model.predict_proba(train_data)[:, 1]
#     prob_flip = model.predict_proba(flip_train_data)[:, 1]
#
#     avg_effect = np.mean(prob_flip - prob)
#     df = data[features].copy()
#     df['deviation'] = (prob_flip - prob) - avg_effect
#     return df

def flip_effect(X_train, y,  flip_attr, classifier):
    model = classifier.fit(X_train, y)
    X_flip = X_train.copy()
    X_flip[flip_attr] = X_flip[flip_attr].apply(lambda x : x ^ 1)

    prob = model.predict_proba(X_train)[:, 1]
    prob_flip = model.predict_proba(X_flip)[:, 1]
    avg_effect = np.mean(prob_flip - prob)

    return (prob_flip - prob) - avg_effect


def numeric_discovery(df, ignore_attr):
    target = ps.NumericTarget('deviation')
    search_space = ps.create_nominal_selectors(df, ignore=['deviation', ignore_attr])
    task = ps.SubgroupDiscoveryTask(df, target, search_space, qf=ps.StandardQFNumeric(1))
    result = ps.BeamSearch().execute(task)

    df_dis = ps.as_df(df, result, statistics_to_show=ps.all_statistics_numeric)
    return df_dis


from encoding import DataEncoding

file_path = '/Users/xiaoqi/Desktop/Master-Thesis/Code/Datasets/adult.csv'
data = read_data(file_path)
data = data.drop(['fnlwgt', 'education-num', 'native-country'], axis=1)
data_new = DataEncoding(data)
df_new = data_new.label_encoding()
X = df_new.drop('target', axis=1)
y = df_new['target']
flip_attr = 'sex'
clf = RandomForestClassifier(random_state=0, n_estimators=10)
deviation = flip_effect(X, y,  flip_attr, clf)

df_deviation = data.copy()
# df_deviation = data_new.discretization()
df_deviation = df_deviation.drop('target', axis=1)
df_deviation['deviation'] = deviation
df_dis = numeric_discovery(df_deviation, flip_attr)

