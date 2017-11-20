#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 02:12:03 2017

@author: rajat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:19:38 2017

@author: rajat
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import sys
import time
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


train = pd.read_csv('train_aWnotuB.csv')
test = pd.read_csv('test_BdBKkAj.csv')

train.dtypes

train['source']= 'train'
test['source'] = 'test'
train=pd.concat([train, test],ignore_index=True)
train.shape


train['ID'] = train['ID'].apply(str)
train.dtypes

train['ID'] = train['ID'].apply(lambda x: x[:-1])

year = lambda x: datetime.datetime.strptime(x, "%Y%m%d%H").year
train['year'] = train['ID'].map(year)
day_of_week = lambda x: datetime.datetime.strptime(x, "%Y%m%d%H" ).weekday()
train['day_of_week'] = train['ID'].map(day_of_week)

weekend = []
for day in train['day_of_week']:
    if int(day) in range(0,4):
        is_weekend = 0
        weekend.append(is_weekend)
    else:
        is_weekend = 1
        weekend.append(is_weekend)
        
train['is_weekend'] = weekend        

month = lambda x: datetime.datetime.strptime(x, "%Y%m%d%H" ).month
train['month'] = train['ID'].map(month)

hour = lambda x: datetime.datetime.strptime(x, "%Y%m%d%H" ).hour
train['hour'] = train['ID'].map(hour)

seasons = [0,0,1,1,1,2,2,2,3,3,3,0] #dec - feb is winter, then spring, summer, fall etc
season = lambda x: seasons[(datetime.datetime.strptime(x, "%Y%m%d%H" ).month-1)]
train['season'] = train['ID'].map(season)

# sleep: 12-6, 6-9: breakfast, 10-14: lunch, 14-17: dinner prep, 17-21: dinner, 21-23: deserts!
times_of_day = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5 ]
from itertools import groupby
[len(list(group)) for key, group in groupby(times_of_day)]
time_of_day = lambda x: times_of_day[datetime.datetime.strptime(x, "%Y%m%d%H").hour]
train['time_of_day'] = train['ID'].map(time_of_day)

#daycount
train['ID']=train['ID'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S"))
train["Date"] = (pd.to_datetime(train["ID"], format="%Y-%m-%d %H:%M"))
train["DayCount"] = train["Date"].apply(lambda x: x.toordinal()/730000)


train_X=train[train['source']=='train']
test_X=train[train['source']=='test']
train_y=train_X['Vehicles']

train_X["year"] = train_X["year"].astype('category')
train_X["year_cat"] = train_X["year"].cat.codes
test_X['year_cat'] = 2

train_X = train_X.drop(['ID','Vehicles', 'DateTime', 'source','Date'], axis=1)
test_X = test_X.drop(['ID','Vehicles', 'DateTime', 'source','Date'], axis=1)
train_X = train_X.drop(['year'], axis=1)
test_X = test_X.drop(['year'], axis=1)

train.to_csv('train_basic.csv',index=False)
test.to_csv('test_basic.csv',index=False)

df_1 = train_X[14592:24840]
df_1['Junction'] = 4
df_2 = train_X[43776:]

train_X=train_X[:43776]
train_X=pd.concat([train_X, df_1], ignore_index=True)
train_X[14592:24840]['Junction']=2
train_X=pd.concat([train_X, df_2], ignore_index=True)


y_1 = train_y[14592:24840]
y_2 = train_y[29184:39432]
y_j4 = pd.DataFrame({"y1":y_1})
y_j4['mean'] = y_j4.assign(y2 = y_2.values).mean(axis=1)
y_j4['mean'] = y_j4['mean'].astype(int)
y_final_prev = (y_j4['mean']/2).astype(int) + 1

y_j4_after = train_y[43776:]

train_y = train_y[:43776]
train_y = pd.concat([train_y, y_final_prev], ignore_index=True)
train_y = pd.concat([train_y, y_j4_after], ignore_index=True)

final_list = [x for x in train_y if (x > 150)]
train_y.pop(40723)
train_X=train_X.drop(train_X.index[40723])

'''
def runXGB(train_X, train_y, test_X, test_y=None):
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.02
        params["min_child_weight"] = 8
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.8
        params["silent"] = 1
        params["max_depth"] = 8
        params["seed"] = 1
        plst = list(params.items())
        num_rounds = 500

        xgtrain = xgb.DMatrix(train_X, label=train_y)
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        pred_test_y = model.predict(xgtest)
        return pred_test_y

preds = runXGB(np.array(train_X), train_y, np.array(test_X))
'''

# Modeling
dtrain = xgb.DMatrix(train_X, label=train_y, missing=np.nan)

param = {'objective': 'reg:linear', 'booster': 'gbtree', 'silent': 1,
		 'max_depth': 7, 'eta': 0.01, 'nthread': 4,
		 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 6,
		 'max_delta_step': 0, 'gamma': 0}
num_round = 690

xgb.cv(param, dtrain, num_round, nfold=4, seed=2244)
exit()
# [690]   cv-test-rmse:2487.3809205+9.82125332763 - 10 690 and 20 - produt category 2,3 removed - v3

seeds = [1122, 2244, 3366, 4488, 5500]
test_preds = np.zeros((len(test_X), len(seeds)))

for run in range(len(seeds)):
	sys.stdout.write("\rXGB RUN:{}/{}".format(run+1, len(seeds)))
	sys.stdout.flush()
	param['seed'] = seeds[run]
	clf = xgb.train(param, dtrain, num_round)
	dtest = xgb.DMatrix(test_X, missing=np.nan)
	test_preds[:, run] = clf.predict(dtest)

test_preds = np.mean(test_preds, axis=1)

submission_final = pd.DataFrame({"ID":test["ID"]})
submission_final["Vehicles"] = test_preds

submission_final.to_csv('submission_final.csv', index=True)

  