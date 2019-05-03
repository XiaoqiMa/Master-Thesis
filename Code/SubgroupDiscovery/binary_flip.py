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


# def train_model(data, features, target, classifier):
#     train_data = data[[features]]
#     target_data = data[target]
#     model = classifier.fit(train_data.values, target_data.values.ravel())
#     return model
#
# def flip_value(data, attr):
#
#     data[attr] = pd.get_dummies(data[attr], drop_first=True)
#     data[attr] = data[attr].apply(lambda x: x ^ 1)
#     return data

def effect_change(data, features, attr, target, classifier):
    train_data = pd.get_dummies(data[features], drop_first=True)
    target_data = data[target]
    df_attr = pd.get_dummies(data[attr], drop_first=True)
    col_name = df_attr.columns[0]
    t_index = df_attr.loc[df_attr[col_name] == True].index
    f_index = df_attr.loc[df_attr[col_name] == False].index

    flip_data = train_data.copy()
    train_data = train_data.join(df_attr, how='inner')
    flip_attr = df_attr.apply(lambda x: x ^ 1)

    flip_train_data = flip_data.join(flip_attr, how='inner')
    model = classifier.fit(train_data.values, target_data.values.ravel())
    prob = np.amax(model.predict_proba(train_data), axis=1)
    prob_flip = np.amax(model.predict_proba(flip_train_data), axis=1)

    diff = np.mean(prob[t_index]) - np.mean(prob[f_index])
    flip_diff = np.mean(prob_flip[f_index]) - np.mean(prob_flip[t_index])

    avg_effect = flip_diff - diff

    df = data[features].copy()
    df['deviation'] = (prob_flip - prob) - avg_effect
    # print(df.head())
    return df


def numeric_discovery(df):
    target = ps.NumericTarget('deviation')
    search_space = ps.create_nominal_selectors(df, ignore='deviation')
    task = ps.SubgroupDiscoveryTask(df, target, search_space, qf=ps.StandardQFNumeric(1))
    result = ps.BeamSearch().execute(task)

    df_dis = ps.as_df(df, result, statistics_to_show=ps.all_statistics_numeric)
    return df_dis


#
file_path = '/Users/xiaoqi/Desktop/Master-Thesis/Code/Datasets/adult.csv'
data = read_data(file_path)
features = ['education', 'occupation', 'race', 'marital-status']
target = 'target'
attr = 'sex'
classifier = RandomForestClassifier(random_state=0)
df = effect_change(data, features, attr, target, classifier)
df_dis = numeric_discovery(df)
df_dis.to_csv('adult_result.csv')
