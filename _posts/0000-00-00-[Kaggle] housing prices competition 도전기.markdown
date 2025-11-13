---
layout: post
title:  "[Kaggle] housing prices competition 도전기"
date: 2025-11-13 19:45:24 +0900
categories: AI ML Kaggle XGBoost 2학년
---
제목을 눌러 본문을 확인하세요.

# Goal
---
It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 

Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)  

The file should contain a header and have the following format:
```
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
etc.
```

# Explanation
---
1461개의 행과 81개의 열을 가진 첫 초대형 데이터 셋을 가지고  
y_test를 예측하는 도전기가 시작된다!
## Data preprocessing
### Import data

```py
import pandas as pd
import numpy as np

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

df = pd.concat([df_train, df_test], axis = 0)

XY = df.iloc[:, 1:-3]
X_test = df_train.iloc[:, -1]

min_valid = len(XY) * 0.7 #행의 길이 * 0.7
XY = XY.dropna(axis=1, thresh=min_valid)

XY = XY.fillna(df.mean(numeric_only=True))
```
train.csv와 test.csv를 불러들인 후, `concat()` 함수로 이 둘을 행을 기준으로 합쳐준다.  
`XY` 변수엔 X_train, y_train에 해당하는 모든 독립변수를 담았고,  
y_test에 대한 정보는 없으므로 df_train에 있는 종속 변수만 `X_test`에 넣어주었다.

독립 변수 열의 약 70%에 해당하는 값이 NA라면 해당 열은 과감히 삭제한다.  
NA가 과하게 많은 경우 y_test의 값을 잘 못 예측할 가능성이 높아진다.

남은 NA의 경우 해당 열 중 오직 int/float 형만 있는 열만 평균을 내어 결측값을 채워준다.

### OneHotEncoding, Feature Scaling
```py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

categorical = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
    'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive'
]

numerical = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
]

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), categorical),
        ('num_scaler', StandardScaler(), numerical)
    ], 
    remainder='passthrough'
)

XY = ct.fit_transform(XY).toarray()
```
OneHotEncoding할 열과 Feature Scaling할 열이 너무 띄엄띄엄이라   
Gemini한테 이걸 어떻게 효율적으로 코드를 작성할 수 있을까 물어봤는데  
저렇게 일일이 열 이름을 리스트화하여 각각 인코더와 스칼러에게 제공하는 게  
제일 안전하고 좋은 방법이라는 충격적인 답변을 받았다.

나보단 Gemini가 똑똑하니까 저 말을 곧이그대로 받아들여 위와 같이  
OneHotEncoding과 Feature Scaling을 진행하였다.

### Splitting into the training set and the data set
```py
X_train = XY[:1460, :]
y_train = XY[1460:, :]
```

## Training the XGBoost model
```py
# from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor

# regressor = SVR(kernel='rbf')
regressor = XGBRegressor(n_estimators=500, learning_rate = 0.05, random_state = 0) # 최고성능
regressor.fit(X_train, X_test)
```
처음엔 SVM-R로 예측을 진행했으나 성능이 좋지 않아  
XGBoost 모델을 사용하였다.

매개변수로 어느 정도 규제를 주는 것이 좋은데,  
`n_estimators` 수를 최대한 많이 주고(앙상블 구성 모델을 많이 돌림)  
`learning_rate` 수를 0.05와 같이 작은 값으로 설정하여 
모델이 오류 수정 시 가중치 업데이트 크기를 작게 제한한다.  

해당 방식이 지금껏 제출한 모델 중 최고 성능을 자랑했다.

## Preparing predicted data file for submission
```py
y = regressor.predict(y_train).reshape(-1, 1)
iid = np.array([x for x in range(1461, len(y)+1461)]).reshape(-1, 1)
y_pred = np.concatenate([iid, y], axis=1)

df_pred = pd.DataFrame(y_pred, columns=['Id', 'SalePrice']) 
# np.array를 dataFrame으로 변환

df_pred['Id'] = df_pred['Id'].astype('int32')
df_pred.to_csv('pred.csv', index=False) # csv 파일 추출
```

# Ranking
---
**326**/5,376  
2025-11-13 20:27:56 +0900