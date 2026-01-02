---
layout: post
title:  "[Kaggle] Titanic - Machine Learning from Disaster 도전기"
date: 2025-11-14 12:27:50 +0900
categories: AI ML Kaggle 2학년
---
제목을 눌러 본문을 확인하세요.

# Goal
---
It is your job to predict if a passenger survived the sinking of the Titanic or not.
For each in the test set, you must predict a 0 or 1 value for the variable.

You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.

The file should have exactly 2 columns:
```
PassengerId (sorted in any order)
Survived (contains your binary predictions: 1 for survived, 0 for deceased)
PassengerId,Survived
892,0
893,1
894,0
Etc.
```

# 1st Explanation
---
## Data preprocessing
### Import data

```py
import numpy as np
import pandas as pd
import sys

df_train = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')

df_train = df_train.dropna(axis=1, thresh=(len(df_train) * 0.7))
df_test = df_test.dropna(axis=1, thresh=(len(df_test) * 0.7))

df_train = df_train.drop(['Ticket', 'Name'], axis=1)
df_test = df_test.drop(['Ticket', 'Name'], axis=1)

df_train = df_train.fillna(df_train.mean(numeric_only=True))
df_test = df_test.fillna(df_test.mean(numeric_only=True))

Xy = pd.concat([df_train.iloc[:, 2:], df_test.iloc[:, 1:]], axis=0)
X_test = df_train.iloc[:, 1]
```
dataset에서 Ticket열(티켓 번호 데이터)과 Name열(승객 이름)은 Servived 여부에 대한  
예측에 방해가 되는 요소에 해당하므로 과감없이 drop 처리한다.  
Age 열 또한 Servived 여부에 관련이 있을까 싶어 해당 열도 빼고 예측을 진행해보았으나,  
accuary가 꽤나 많이 감소하게 되어 Ticket 열과 Name 열만 drop 처리하기로 결정하였다.

### OneHotEncoding, Feature Scaling
```py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

categorical = ['Sex', 'Embarked']

numerical = ['Pclass', 'Age', 'SibSp', 'Parch']

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical), ('num_scaler', StandardScaler(), numerical)], remainder='passthrough')
Xy = ct.fit_transform(Xy)
```
XGBoost로 classify하기 위해 범주형 데이터와 numeric 데이터를  
각각 OneHotEncoding, Feature Scaling 처리하였다.

### Splitting into the training set and the data set
```py
X_train = Xy[:891, :]
y_train = Xy[891:, :]
```

## Training the XGBoost model
```py
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC

svml = SVC(kernel='rbf', random_state = 0)
xgb = XGBClassifier(n_estimators=700, learning_rate = 0.1, random_state = 0, max_depth = 9)

print(X_train)
xgb.fit(X_train, X_test)
```

해당 방식이 지금껏 제출한 모델 중 최고 성능을 자랑했다.

## Preparing predicted data file for submission
```py
X_pred = xgb.predict(X_train)
iid = np.array([x for x in range(892, len(X_pred)+892)])

submission_data = np.concatenate([iid.reshape(-1, 1), X_pred.reshape(-1, 1)], axis = 1)

np.set_printoptions(threshold=sys.maxsize)
submission_data = pd.DataFrame(submission_data, columns=['PassengerID', 'Survived'])
                               
submission_data.to_csv('./answer/pred.csv', index=False)
```

## Ranking
**12,592**/16,529  
2025-11-13 20:27:56 +0900

# 2nd Explanation
---
## Data preprocessing
### OneHotEncoding, Feature Scaling
