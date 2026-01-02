---
layout: post
title:  "[Kaggle] Top 2%의 방법으로 Titanic 문제 풀어보기"
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

# Data Analysis
---
먼저 원활한 데이터 분석을 위해 각 열의 데이터 타입을 구분해보았다.
```
Survived: Survival (0 = No; 1 = Yes) => Binary Variable
Pclass: Ticket class (1 = 1st; 2 = 2nd; 3 = 3rd) => Ordinal Variable
Sex: Sex => Nominal Variable / Binary Variable
Age: Age in years => Continuous Variable
SibSp: # of siblings / spouses aboard the Titanic => Discrete Variable
Parch: # of parents / children aboard the Titanic => Discrete Variable
Ticket: Ticket number => Nominal Variable
Fare: Passenger fare => Continuous Variable
Cabin: Cabin number => Nominal Variable
Embarked : Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton) => Nominal Variable
```
```py
import numpy as np
import pandas as pd

df_train = pd.read_csv('./dataset/train.csv', index_col = 0)
df_test = pd.read_csv('./dataset/test.csv', index_col = 0)

X = pd.concat([df_train.drop(['Survived'], axis=1), df_test], axis=0)
y = df_train['Survived']

X.info()
```
```py 
numeric = X.select_dtypes(exclude=['object']).drop(['Pclass'], axis=1).copy()

numeric.columns
```
```py
continuous = ['Age', 'Fare']
discrete = []

for i in numeric.columns:
    if i not in continuous:
        discrete.append(i)
```
```py
categorical = X.select_dtypes(include=['object']).copy()
categorical['Pclass'] = X['Pclass']

categorical.columns
```