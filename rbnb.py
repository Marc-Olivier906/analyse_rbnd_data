#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:53:30 2021

@author: Marc-Olivier
"""
import pandas as pd

listing = pd.read_csv('/Users/Marc-Olivier/Desktop/data_rbnb/listings.csv')
reviews = pd.read_csv('/Users/Marc-Olivier/Desktop/data_rbnb/reviews.csv')
calendar = pd.read_csv('/Users/Marc-Olivier/Desktop/data_rbnb/calendar.csv')
listing2 = pd.read_csv('/Users/Marc-Olivier/Desktop/data_rbnb/listings-2.csv')
reviews2 = pd.read_csv('/Users/Marc-Olivier/Desktop/data_rbnb/reviews-2.csv')
voisin = pd.read_csv('/Users/Marc-Olivier/Desktop/data_rbnb/neighbourhoods.csv')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import warnings
#%matplotlib inline
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)

calendar.head(5)

calendar['available'] = calendar.available.map(lambda x: 1 if x == 't' else 0)
calendar.date = pd.to_datetime(calendar.date)
calendar['price'] = calendar['price'].str.replace('$', '').str.replace(',', '')
calendar['price'] = calendar['price'].astype(float)

calendar_annee = calendar.groupby('date')['available', 'price'].mean().reset_index()
calendar_annee.rename(columns={'price': 'average_price', 'available': 'vacancy'}, inplace=True)
calendar_annee['occupancy'] = (1 - calendar_annee['vacancy']) * 100
calendar_annee['dayofweek'] = calendar_annee.date.dt.weekday_name.str[:3]
calendar_annee['month'] = calendar_annee.date.dt.month_name().str[:3]

def plot_calendar(groupby_col, agg_col):
    df_index = list(calendar_annee[groupby_col].unique())
    grouped_df = calendar_annee.groupby(
        groupby_col)[agg_col].mean().reindex(df_index)

    plt.plot(grouped_df)
    plt.ylabel(agg_col.replace('_', ' ').title())
    plt.title(" {} by {}".format(agg_col.replace('_', ' ').title(),groupby_col.title()), fontsize=18,fontweight='bold')
    ticks = list(range(len(df_index)))
    labels = df_index
    plt.xticks(ticks, labels)
    plt.show()
    
plot_calendar('month', 'occupancy')
plot_calendar('month', 'average_price')
plot_calendar('dayofweek', 'occupancy')
plot_calendar('dayofweek', 'average_price')


features = ['accommodates', 'reviews_per_month', 'price', 'review_scores_rating', 'number_of_reviews','minimum_nights']

data = listing[features]
num_cols = data.select_dtypes(exclude='object').columns
cat_cols = data.select_dtypes(include='object').columns
data[cat_cols] = data[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))
data[num_cols] = data[num_cols].apply(lambda col: col.fillna(col.median()))

data[data.columns[1:]] = data[data.columns[1:]].replace('[\$,]', '', regex=True).astype(float)

def train_and_test(df, model, test_size=0.2):
    target = df['price']
    features = df.copy().drop('price', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test) 
    print('R^2 test: %.3f' % (r2_score(y_test, y_pred)))
    pred = np.round(np.exp(y_pred) + 1, 1)
    actual = np.round(np.exp(y_test) + 1, 1)
    plt.scatter(actual.as_matrix(), pred)
    plt.title('Predicted vs. Actual Price', fontsize=18, fontweight='bold')
    plt.xlabel('Actual Listing Price')
    plt.ylabel('Predicted Listing Price')
    plt.show()
    return model

model = LGBMRegressor()
trained_model = train_and_test(data, model)


feat_imp = pd.Series(trained_model.feature_importances_,index=data.columns.drop('price'))
feat_imp.nlargest(15).plot(kind='barh', figsize=(10, 6))
plt.xlabel('Relative Importance')
plt.title("Feature importances", fontsize=18, fontweight='bold')
plt.show()


