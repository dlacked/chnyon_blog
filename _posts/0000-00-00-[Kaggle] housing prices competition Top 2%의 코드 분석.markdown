---
layout: post
title:  "[Kaggle] housing prices competition Top 2%의 코드 분석(나는 아님)"
date: 2025-11-27 13:29:33 +0900
categories: AI ML Kaggle 2학년
---
제목을 눌러 본문을 확인하세요.

해당 게시물은 아래 게시물을 참조하였으며,  
일부 내용을 추가하였습니다.  
https://mac520.tistory.com/3  
https://mac520.tistory.com/5  
https://mac520.tistory.com/8  
https://mac520.tistory.com/9  

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

# Data Analysis
---
```py
import pandas as pd
import numpy as np

df_train = pd.read_csv('./data/train.csv', index_col = 0)
df_test = pd.read_csv('./data/test.csv', index_col = 0)

X = pd.concat([df_train.drop("SalePrice", axis=1), df_test], axis=0);
y = df_train[['SalePrice']]

X.info()
```
train.csv와 test.csv에서 ID 변수를 index로 지정한 채로  
csv를 읽어들인다.

df_train에서 SalePrice(종속 변수)를 제외한 모든 변수와   
df_test의 모든 변수를 행을 기준으로 이어준다.

```
<class 'pandas.core.frame.DataFrame'>
Index: 2919 entries, 1 to 2919
Data columns (total 79 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   MSSubClass     2919 non-null   int64  
 1   MSZoning       2915 non-null   object 
 2   LotFrontage    2433 non-null   float64
 ...
 76  YrSold         2919 non-null   int64  
 77  SaleType       2918 non-null   object 
 78  SaleCondition  2919 non-null   object 
dtypes: float64(11), int64(25), object(43)
memory usage: 1.8+ MB
```

X.info()를 통해 SalePrice 변수가 잘 삭제되었고,  
concat이 원하는 대로 진행되었음을 확인할 수 있다.

## Numerical Variables
```py
numeric = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()
numeric.columns
```
독립 변수의 집합 `X`에서 숫자형 데이터를 추출하기 위해   
DataFrame에서 데이터 타입이 object인 열을 제외하도록 설정한다.  
즉, Categorical Variables를 제외한 나머지 열들만 추출한다.

MSSubClass는 숫자형 데이터에 해당하나  
사실상 '건물 종류'를 구분하는 범주형 변수에 해당하므로  
object 리스트에 추가하는 것이 바람직하다.

## Splitting into Continuous and Discrete Variables
```py
disc_num_var = ['OverallQual','OverallCond', 'BsmtFullBath', 
                'BsmtHalfBath','FullBath','HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
                'GarageCars', 'MoSold', 'YrSold']

cont_num_var = []
for i in numeric.columns:
    if i not in disc_num_var:
        cont_num_var.append(i)
```
숫자형 변수 numeric 중 이산형 변수와 연속형 변수를 구분한다.  
모두 int64에 해당하나 이산형 변수의 경우 의미상으로는 count 또는 등급 등  
'구분 가능한 값'으로 사용되는 변수다.

블로그에는 추가하지 않은 이산형 변수
'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'를 추가로  
`disc_num_var`에 추가해보았다.

## Categorical Variables
```py
categorical = X.select_dtypes(include=['object']).copy()
categorical['MSSubClass'] = X['MSSubClass']
categorical.columns
```
**MSSubClass는 이산형 변수에 속하면 안 되는 것인가?**  
안 된다. 범주형 변수는 그룹이나 종류로 나누기 위해 사용한다.   
이는 값이 숫자로 되어 있더라도 그 숫자는 단지 구분자의 역할을 할 뿐이다.

하지만 이산형 변수는 숫자 자체에 크기, 순서, 혹은 횟수에 대한 의미를 가진다.  
일반적으로 countable한 값을 나타내는 것이다.

MSSubClass는 단순히 건물의 종류를 나타내는 **코드**일 뿐이므로,  
범주형 변수 집합에 속하는 것이 바람직하다.

# Visualize
---
## Continuous Variable Value Distribution
```py
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

fig = plt.figure(figsize=(18, 16))
for index, col in enumerate(cont_num_var):
    plt.subplot(6, 4, index+1) # subplot 위치 정하기
    sns.histplot(numeric.loc[:, col].dropna(), kde=False)
    # numeric 중 NA 값은 추가하지 않음
    # x열은 col로 지정

fig.tight_layout(pad=1) #레이아웃 조정
```
숫자형 변수의 value 분포를 확인하기 위한 히스토그램을 출력한다.

분포에 0이 많은 변수는 예측 시 실제로 정보가 추가되지 않는다.    
해당 변수는 0의 값을 대부분으로 가지므로 처리해 줄 필요가 있다.  

해당 조건에 해당하는 변수는 다음과 같다.  
BsmtFinSF2, LowQualFinSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal

## Continuous Variable Outlier Distribution
```py
fig = plt.figure(figsize=(14, 15))
for index, col in enumerate(cont_num_var):
    plt.subplot(6, 4, index+1)
    sns.boxplot(x=col, data = numeric.dropna())
fig.tight_layout(pad=1)
```
연속형 변수의 value 분포를 확인하기 위한 box plot을 출력한다.

대부분이 0인 변수는 0이 아닌 나머지 값들이 모두 이상치로 처리된다.  
분포에 0이 많은 변수는 제외하고, 그 외 변수들 중  
유난히 이상값이 심한 변수는 추후 작업이 필요하다.

해당 조건에 해당하는 변수는 다음과 같다.  
LotFrontage, LotArea, BsmtFinSF1, TotalBsmtSF, GrLivArea  

## Discrete Variable Distribution
```py
fig = plt.figure(figsize=(20, 15))
for index, col in enumerate(disc_num_var):
    plt.subplot(5, 3, index+1)
    sns.countplot(x=col, data=numeric.dropna(), palette='Accent', hue=col, legend=False) 
fig.tight_layout(pad=1)
```
이산형 변수의 value 분포를 확인하기 위한 막대그래프를 출력한다.

대부분의 변수의 값 분포가 고른 편이라는 것을 확인할 수 있다.

## Categorical Variable Distribution
```py
fig = plt.figure(figsize=(18, 20))
for index in range(len(categorical.columns)):
    plt.subplot(9, 5, index+1)
    sns.countplot(x=categorical.iloc[:, index], data = categorical.dropna(), palette='Accent', hue=categorical.iloc[:, index], legend=False)
    plt.xticks(rotation=90)
fig.tight_layout(pad=1)
```
범주형 변수의 value 분포를 확인하기 위한 막대그래프를 출력한다.

변수의 값 분포가 오직 하나의 값으로만 구성되어 있는 경우 유용한 정보를 제공하지 못한다.  
해당 조건에 해당하는 변수는 다음과 같다.

Street, Utilities, Condition2, RoofMatl, Heating, (Functional)

# Bivariate Analysis
---
## Correlation Matrix
2개의 다른 특징을 살펴보고 해당 특징 간 가능한 관계나 독튼한 패턴을 식별한다.  
일반적으로 사용되는 기술은 Numerical Variables를 대상으로  
Correlation Matrix를 사용하는 것이다.

Correlation Matrix는 2개의 연속적인 특징 간의 선형(상관) 관계를 발견하는 효과적인 도구다.  
상관 관계는 종속 변수에 중요한 특징을 결정할 수 있을 뿐 아니라  
독립적인 예측 변수 간의 다중 공선성을 조사하는 수단으로 사용할 수도 있다.  

다중 공선성은 예측 모델 구성 시 제거하는 방법이 일반적이다.
```py
plt.figure(figsize=(14, 12))
correlation = numeric.corr() # numeric의 상관관계 행렬 저장(pearson)
sns.heatmap(correlation, mask=correlation < 0.8, linewidth=0.5, cmap='Blues')
```
GarageCars와 GarageArea, GarageYrBlt와 YearBuilt, TotRmsAbvGrd와 GrLivArea,  
1StFlrSF와 TotalBsmtSF 순으로 높은 상관관계를 가지고 있다.

이들은 다중 공선성을 야기할 수 있어 제거해주는 편이 좋다.

## Correlation with SalePrice(Target Column)
```py
numeric_train = df_train.select_dtypes(exclude=['object'])
# SalePrice가 포함된 train 데이터에서 숫자형 변수만 추출

correlation = numeric_train.corr()
correlation[['SalePrice']].sort_values(['SalePrice'], ascending=False).head(10)
```
SalePrice(종속 변수)와의 높은 상관 관계를 가진 상위 10개의 숫자형 변수 column을 출력한다.

OverallQual과 GrLivArea 변수가 종속 변수와의 상관 관계 중 Top 2에 속한다.  
따라서 이 두 변수는 종속 변수 값을 예측하는 데 중요한 요소가 될 것이다. 

## Scatter Visualization
산점도 시각화 시 수치적 특성 간의 잠재적 선형 관계를 식별하는 데 도움이 될 수 있다.  
이는 Correlation으로 포함되지 않는 관계까지 포함한 2차/지수 관계 등을 확인할 수 있다.
```py
fig = plt.figure(figsize=(20, 20))
for index in range(len(numeric_train.columns)):
    plt.subplot(10, 5, index+1)
    sns.scatterplot(x=numeric_train.iloc[:, index], y='SalePrice', data=numeric_train.dropna())
fig.tight_layout(pad=1)
```
독립 변수들 중에서 지수 관계 그래프는 찾기 어려운 것으로 보인다.

# Data Preprocessing
---
## Dropping the Correlation Features
```py
X.drop(['GarageCars', 'GrarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'], axis=1, inplace=True)
```
다중 공선성 문제를 피하기 위해 상관계수가 높은 두 쌍의 Feature 중 하나를 제거한다.

## Dropping the NA Features
```py
plt.figure(figsize=(25, 8))
plt.title('Number of Missing Rows')
missing_count = pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by='sum', ascending=False).head(20).reset_index()
missing_count.columns = ['features', 'sum']
sns.barplot(x='features', y='sum', data=missing_count)
```
독립 변수의 집합에서 각 변수의 총 NA 수를 막대그래프로 보여준다. 

NA가 많은 변수는 예측 시 유용한 정보를 제공하지 않으므로 제거가 필요하다.  
해당 조건에 해당하는 변수는 다음과 같다.  

PoolQC, MiscFeature, Alley
```py
X.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)
```

## Dropping the Linear Associagion Features with SalePrice
Scatter Visualization과 Correlation with SalePrice(Target Column) 결과를 종합했을 때,  
MoSold와 YrSold는 SalePrice와 선형적인 관계가 거의 없음을 확인할 수 있다.

따라서 해당 변수는 모델링 과정에서 불필요한 영향을 줄 수 있으므로 제거하는 것이 바람직하다.
```py
X.drop(['MoSold', 'YrSold'], axis=1, inplace=True)
```

## Dropping Values that Would Cause Overfitting
일부 Feature는 데이터의 거의 모든 값이 동일하며, 모델이 해당 변수를 통해 유의미한 패턴을 학습할 수 없다.  
이러한 변수는 overfitting의 위험을 증가시키고, 모델 성능에 좋은 영향을 미치지 않는다.  
해당 변수를 제거하기 위해 각 변수의 값들이 데이터의 96% 이상을 차지하는지 확인하고,  
해당 조건을 만족하는 변수는 제거한다.
```py
cat_col = X.select_dtypes(include=['object']).columns
overfit_cat = []
for i in cat_col:
    counts = X[i].value_counts()
    if counts.iloc[0] / len(X) * 100 > 96:
        overfit_cat.append(i)

num_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
overfit_num = []
for i in num_col:
    counts = X[i].value_counts()
    if counts.iloc[0] / len(X) * 100 > 96:
        overfit_num.append(i)

print("Categorical Features with >96% of the same value: ", overfit_cat)
print("Numerical Features with >96% of the same value: ", overfit_num)
X.drop(overfit_cat + overfit_num, axis=1, inplace=True)
```

# Dropping the Outliers
---
이상치 처리를 통해 모델이 극단적인 값에 영향을 받지 않도록 한다.

이를 통해 모델의 성능이 안정적이고 정확하게 나올 수 있도록 한다.

## Dropping the Outliers over the Threshold
Sequential Variable Outlier Distribution을 확인하면  
기준점을 심하게 뛰어넘은 이상치들이 존재한다.

LotFrontage, LotArea, BsmtFinSF1, TotalBsmtSF, GrLivArea

threshold를 설정하여 극단적인 값을 기준으로 이상치를 제거한다.
```py
df_train = df_train.drop(df_train[df_train['LotFrontage'] > 200].index)
df_train = df_train.drop(df_train[df_train['LotArea'] > 100000].index)
df_train = df_train.drop(df_train[df_train['BsmtFinSF1'] > 4000].index)
df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 5000].index)
df_train = df_train.drop(df_train[df_train['GrLivArea'] > 4000].index)
```

# Fill the NA Features
---
결측치를 채워 모델의 효율성을 높이고, 예측 정확도를 향상시킨다.  
ML 모델은 누락된 값을 처리할 수 없으나 결측치를 적절하게 처리하여  
기존 데이터의 본질을 유지하면서 모델이 최적의 성능을 낼 수 있도록 도움을 준다.

## Visualizing the NA Features
```py
pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by='sum', ascending=False).head(30)
```
각 변수당 결측치 현황을 DataFrame 형태로 보여준다.

```py
plt.figure(figsize=(25, 8))
plt.title('Number of Missing Rows')
missing_count = pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by='sum', ascending=False).head(15).reset_index()
missing_count.columns = ['features', 'sum']
sns.barplot(x='features', y='sum', data=missing_count)
```

## Replacing the Values of the NA Ordinal Features to "NA"
```py
cat = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1',
       'GarageCond','GarageQual','BsmtCond','BsmtQual','FireplaceQu','Fence',
       "KitchenQual","HeatingQC",'ExterQual','ExterCond'] # The Continuous Features which have NA Value(s)
X[cat] = X[cat].fillna("NA")
```
Ordinal Variables는 값들 사이에 명확한 순서나 등급이 있기 때문에  
결측치 처리 시 평균이나 중앙값으로 대체하는 것은 적절하지 않을 수 있다.  
"NA"라는 새로운 등급을 부여하여 처리하는 것이 분석을 더 의미있게 할 수 있다.

## Replacing the Values of the NA Categorical Features
```py
cols = ['MasVnrType', 'MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Electrical', 'Functional']
pd.set_option('future.no_silent_downcasting', True)

X[cols] = X.groupby('Neighborhood')[cols].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))
```
범주형 피쳐는 그 변수의 최빈값을 사용하는 방법이 일반적이다.  
그러나 단순히 전역적으로 최빈값을 적용하기 보다는, 각 Neighborhood 내에서  
결측치가 있는 변수들의 최빈값을 찾아 결측치를 채우는 방식을 적용해보았다.  
해당 방법은 특정 동네의 특성에 맞게 결측치가 대체되므로 더 의미 있는 데이터가 만들어진다.

## Replacing the Values of the NA Continuous Features
```py
cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", 
        "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]
X[cont] = X[cont].fillna(X[cont].mean())
```
Continuous Features는 보통 평균값으로 대체하는 방식이 가장 일반적이다.  

하지만 어떤 Continuous Features는 Neighborhood마다 값의 차이가 클 수 있다.  
이런 경우, 전체 평균을 사용해 결측치를 대체하면 큰 오차가 발생할 수 있다.

**도메인 지식을 활용하여 데이터를 이해하고 각 변수의 특성에 맞게**  
**결측치를 처리하는 것이 예측 모델의 성능을 크게 향상시킬 수 있다.**

다음은 Neighborhood 값이 무엇이냐에 따라  
LostFrontage와 GarageArea의 값 평균 차이가 큰 모습을 확인할 수 있다.
```py
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
X['GarageArea'] = X.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# 마지막은 MSZoning 변수의 결측치를 각 MSSubClass 별로 최빈값을 사용해서 대체
```
```py
print('Mean of LotFrontage: ', X['LotFrontage'].mean())
print('Mean of GarageArea: ', X['GarageArea'].mean())
```
```
Mean of LotFrontage:  69.96227441551899
Mean of GarageArea:  472.8145396726098
```

각 Neighborhood 값에 따라 두 변수의 값이 얼마나 차이나는지 확인하고자  
다음과 같은 코드를 작성할 수 있다.
```py
neigh_lot = X.groupby('Neighborhood')['LotFrontage'].mean().reset_index(name='LotFrontage_mean')
neigh_garage = X.groupby('Neighborhood')['GarageArea'].mean().reset_index(name='GarageArea_mean')

fig, axes = plt.subplots(1, 2, figsize=(22,8))
axes[0].tick_params(axis='x', rotation=45)
sns.barplot(x='Neighborhood', y='LotFrontage_mean', data=neigh_lot, ax=axes[0], palette='magma', hue='Neighborhood')

axes[1].tick_params(axis='x', rotation=45)
sns.barplot(x='Neighborhood', y='GarageArea_mean', data=neigh_garage, ax=axes[1], palette='magma', hue='Neighborhood')
```

# Changing the Data Type and Mapping the Sequential Features
---
## Changing the Data Type at MSSubClass
```py
X['MSSubClass'] = X['MSSubClass'].apply(str)
```
MSSubClass는 사실성 범주형 feature이므로 문자열로 반환한다.

## Mapping the Sequential Features
순서형 feature는 값들 간 명확한 순서가 존재하는 특성을 가진다.  
피쳐 값들이 정량적 의미를 가지게 되어, 모델이 이들을 숫자적으로 이해하고 예측에 활용할 수 있게 만든다.
```py
ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}

ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']
for col in ord_col:
    X[col] = X[col].map(ordinal_map)

fin_col = ['BsmtFinType1','BsmtFinType2']
for col in fin_col:
    X[col] = X[col].map(fintype_map)

X['BsmtExposure'] = X['BsmtExposure'].map(expose_map)
X['Fence'] = X['Fence'].map(fence_map)
```

# Feature Engineering
---
## Making Additional Features
도메인 지식을 활용하여 주택 가격 예측에 도움을 주는 새로운 Feature를 생성한다.  

예를 들어, LotFrontage와 LotArea를 더한 값을 저장하는 TotalLot을 생성하거나  
TotalBsmtSF와 1stFlrSF, 2ndFlrSF를 합산해 TotalSF를 추가할 수 있다.
```py
X['TotalLot'] = X['LotFrontage'] + X['LotArea']
X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
X['TotalBath'] = X['FullBath'] + X['HalfBath']
X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
```

## Making Binary Value
특징 면적 관련 피처들은 값이 0이면 존재하지 않음, 0보다 크면 존재함을 의미한다.

이러한 특성을 활용해, 각 피쳐의 존재 유무를 나타내는 이진 피쳐를 생성함으로써  
모델이 해당 요소의 유무 자체를 학습할 수 있도록 도와준다.

즉, 새로운 이진 피쳐는 값이 1이면 존재함을 의미하고, 0이면 존재하지 않음을 의미한다.
```py
cols = ['MasVnrArea', 'TotalBsmtFin', 'TotalBsmtSF', '2ndFlrSF', 'WoodDeckSF', 'TotalPorch']

for col in cols:
    X[col + '_bin'] = X[col].apply(lambda x: 1 if x > 0 else 0)
```
이진 피쳐 생성은 피쳐의 유무가 가치를 크게 좌우하는 데이터셋에서 매우 효과적이다.

## Incoding Categorical Features
ML은 범주형 데이터를 직접적으로 처리할 수 없다.  
따라서 문자형 변수는 모델이 이해할 수 있도록 수치형으로 변환해야 한다.  
이 때 가장 널리 사용하는 방법 중 하나가 One-Hot Encoding이다.  

One-Hot Encoding은 각 범주에 고유한 순서를 부여하지 않고 서로 독립된 열로 분리함으로써  
잘못된 순서 정보를 피할 수 있다.
```py
X = pd.get_dummies(X)
```

## Chacking and Changing the Target Distribution
```py
plt.figure(figsize=(10, 6))
plt.title('Before transformation of SalePrice')
sns.histplot(y['SalePrice'], kde=True)
```
차트를 확인하면 꼬리가 우측으로 길게 형성되어 있음을 확인할 수 있다.  
즉 m_3 > 0인 형태의 그래프를 확인할 수 있다.

이럴 땐 로그 변환을 적용해 skewness를 줄이고, variance를 안정화하여  
정규분포와 유사한 형태로 만드는 것이 바람직하다.
```py
plt.figure(figsize=(10, 6))
plt.title('Before transformation of SalePrice')
sns.histplot(np.log(df_train['SalePrice']), kde = True)
```
로그 변환 후 그래프를 확인하면 분포가 훨씬 대칭에 가까워진 것을 확인할 수 있다.  
변환된 값을 종속 변수에 저장하여 학습에 사용할 준비를 마친다.
```py
y['SalePrice'] = np.log(y['SalePrice'])
```

# Modeling and Ensembling
---
## Splitting the Data
```py
x = X.loc[df_train.index]
y = y.loc[df_train.index]
test = X.loc[df_test.index]
```
## Feature Scaling
```py
from sklearn.preprocessing import RobustScaler

cols = x.select_dtypes(np.number).columns # 숫자형 변수만 선택
transformer = RobustScaler().fit(x[cols])
x[cols] = transformer.transform(x[cols])
test[cols] = transformer.transform(test[cols])
```
RobustScaler를 사용하면 평균과 표준편차가 아닌 중간값과 IQR을 기준으로 스케일링을 수행한다.  
이상치에 덜 민감하게 데이터를 변환할 수 있다.

## Splitting into the Traing Set and the Test Set
```py
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
```

## Importing the Models
```py
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
``` 

## XGBoost
```py
xgb = XGBRegressor() 

# 하이퍼파라미터 튜닝

from sklearn.model_selection import RandomizedSearchCV

param_lst = {
    'learning_rate': [0.01, 0.1, 0.15, 0.3, 0.5],
    'n_estimators': [100, 500, 1000, 2000, 3000],
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 5, 10, 20],
    'reg_alpha': [0.001, 0.01, 0.1],
    'reg_lambda': [0.001, 0.01, 0.1]
} 

xgb_reg = RandomizedSearchCV(estimator=xgb, param_distributions=param_lst, n_iter = 100, scoring='neg_root_mean_squared_error', cv=5)
# n_iter번 추출하여 모델을 학습하고 CV를 통해 성능을 평가한다.
# n_iter가지의 조합 중 가장 좋은 성능(가장 높은 neg_root_MSE)을 보인 하이퍼파라미터 조합과 해당 조합으로 학습된 모델을 저장한다.
xgb_search = xgb_reg.fit(X_train, y_train)

# 최적 하이퍼파라미터 적용

best_param = xgb_search.best_params_
xgb = XGBRegressor(**best_param)
```
**하이퍼파라미터 튜닝**  

모델 성능을 최적화하기 위해 여러 하이퍼파라미터를 랜덤 서치(RandomizedSearchCV)를 사용해 튜닝한다.  
랜덤 서치는 주어진 하이퍼파라미터 공간에서 임의로 조합을 선택하여 최적의 파라미터를 찾는 방법이다.

**최적 하이퍼파라미터 적용**

랜덤 서치 후, 최적의 하이퍼파라미터를 선택하여 모델을 재정의한다.  
`best_params_`를 통해 최적의 하이퍼파라미터를 얻고, 이를 기반으로 모델을 업데이트한다.

## LightGBM
```py
lgbm = LGBMRegressor(n_jobs=-1, verbose=-1)
# 경고를 줄이기 위한 n_jobs=-1
# verbose=-1 출력되는 메세지 최소화

param_lst = {
    'max_depth': [2, 5, 8, 10],
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'n_estimators': [100, 300, 500, 1000, 1500],
    'lambda_l1': [0.0001, 0.001, 0.01],
    'lambda_l2': [0, 0.0001, 0.001, 0.01],
    'feature_fraction': [0.4, 0.6, 0.8],
    'min_child_samples': [5, 10, 20, 25]
}

lightgbm = RandomizedSearchCV(estimator=lgbm, param_distributions=param_lst, n_iter = 100, scoring='neg_root_mean_squared_error', cv=5)
lightgbm_search = lightgbm.fit(X_train, y_train)

best_param = lightgbm_search.best_params_
lgbm = LGBMRegressor(**best_param)
```

## CatBoost
```py
cb = CatBoostRegressor(loss_function='RMSE', logging_level='Silent')

# 하이퍼파리미터 튜닝
param_lst = {
    'n_estimators': [100, 300, 500, 1000, 1300, 1600],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'l2_leaf_reg': [0.001, 0.01, 0.1],
    'random_strength': [0.25, 0.5, 1],
    'max_depth': [3, 6, 9],
    'min_child_samples': [2, 5, 10, 15, 20],
    'rsm': [0.5, 0.7, 0.9]
}

catboost = RandomizedSearchCV(estimator=cb, param_distributions=param_lst, n_iter=100, scoring='neg_root_mean_squared_error', cv=5)
catboost_search = catboost.fit(X_train, y_train)

best_param = catboost_search.best_params_
cb = CatBoostRegressor(**best_param)
```

## Evaluating the Models
각 모델을 training set 또는 validation set에서 평가하고,  
MAE(Mean Absolute Error), RMSE(Root Mean Squared Error) 지표를 사용하여 성능을 비교한다.
```py
def mean_cross_val(model, X, y):
    score=cross_val_score(model, X, y, cv=5)
    return score.mean()

# CatBoost
cb.fit(X_train, y_train)
preds_cb = cb.predict(X_val) # pred
mae_cb = mean_absolute_error(y_val, preds_cb) # MAE
rmse_cb = np.sqrt(mean_squared_error(y_val, preds_cb)) # RMSE
cv_cb = mean_cross_val(cb, x, y) # CV

# XGBoost
xgb.fit(X_train, y_train)
preds_xgb = xgb.predict(X_val) # pred
mae_xgb = mean_absolute_error(y_val, preds_xgb) # MAE
rmse_xgb = np.sqrt(mean_absolute_error(y_val, preds_xgb)) # RMSE
cv_xgb = mean_cross_val(xgb, x, y) # CV

# LightGBM
lgbm.fit(X_train, y_train)
preds_lgbm = lgbm.predict(X_val) # pred
mae_lgbm = mean_absolute_error(y_val, preds_lgbm) # MAE
rmse_lgbm = np.sqrt(mean_absolute_error(y_val, preds_lgbm)) #RMSE
cv_lgbm = mean_cross_val(lgbm, x, y) # CV
```
**MAE**  
MAE는 예측값과 실제값 간의 절대적인 차이를 평균한 값이다.  
큰 오차에 대해 선형적으로 반응하여 이상치에 덜 민감하다.  
하지만, 큰 오차를 강조하지 않으므로 이상치에 대한 모델의 민감도를 반영하지 못한다.

**RMSE**  
RMSE는 MSE를 구하고, 그 값을 제곱근한 값이다.  
예측 오차의 크기를 제곱하여 강조하므로, 큰 오차가 있을 때 모델이 이를 더 신경 쓰게 만든다.  
다만 이상치에 고도하게 민감할 수 있다.  

**CV**  
CV는 데이터를 여러 번 나누어 모델을 학습하고 평가하는 방법이다.   
여기서는 5-fold CV를 사용하여 모델을 평가하고, 그 결과의 평균을 구해 CV 점수로 나타낸다.  
훈련 데이터에 과접합된 모델을 방지할 수 있으나 계산량이 많아 시간이 많이 소요된다.

```py
model_performances = pd.DataFrame({
    'Model': ['XGBoost', 'LGBM', 'CatBoost'],
    'CV(S)': [cv_xgb, cv_lgbm, cv_cb],
    'MAE': [mae_xgb, mae_lgbm, mae_cb],
    'RMSE': [rmse_xgb, rmse_lgbm, rmse_cb]
})

print("Sorted by RMSE:")
print(model_performances.sort_values(by='RMSE'))
```
모델 성능을 DataFrame 형태로 정리하여 비교한다.
CatBoost 모델이 가장 뛰어난 성능을 보이는 것으로 확인된다.

## Blending
```py
def blend_models_predict(X, b, c, d):
    return (b * xgb.predict(X) + c * lgbm.predict(X) + d * cb.predict(X))

subm = np.exp(blend_models_predict(test, 0.4, 0.3, 0.3))
```
여러 모델의 예측값을 가중 평균하여 최종 예측값을 생성하는 앙상블 기법이다.

## Submission
```py
submission = pd.DataFrame({'ID': test.index, 'SalePrice': subm})
submission.to_csv('submission.csv', index=False)
```