#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:14:12 2018

@author: ryan
"""
###############################################################################
# Libraries
###############################################################################
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

###############################################################################
# load the data
###############################################################################
train_df = pd.read_csv('~/kaggle/house-prices/data/train.csv')
test_df = pd.read_csv('~/kaggle/house-prices/data/test.csv')
###############################################################################
# categorize features
###############################################################################
backward = {'MSSubClass':'Nominal', 'MSZoning':'Nominal', 'LotFrontage':'Continuous', 'LotArea':'Continuous', 'Street':'Nominal', 'Alley':'Nominal', 'LotShape':'Ordinal', 'LandContour':'Nominal', 'Utilities':'Ordinal', 'LotConfig':'Nominal', 'LandSlope':'Ordinal', 'Neighborhood':'Nominal', 'Condition1':'Nominal', 'Condition2':'Nominal', 'BldgType':'Nominal', 'HouseStyle':'Nominal', 'OverallQual':'Ordinal', 'OverallCond':'Ordinal', 'YearBuilt':'Discrete', 'YearRemodAdd':'Discrete', 'RoofStyle':'Nominal', 'RoofMatl':'Nominal', 'Exterior1st':'Nominal', 'Exterior2nd':'Nominal', 'MasVnrType':'Nominal', 'MasVnrArea':'Continuous', 'ExterQual':'Ordinal', 'ExterCond':'Ordinal', 'Foundation':'Nominal', 'BsmtQual':'Ordinal', 'BsmtCond':'Ordinal', 'BsmtExposure':'Ordinal', 'BsmtFinType1':'Ordinal', 'BsmtFinSF1':'Continuous', 'BsmtFinType2':'Ordinal', 'BsmtFinSF2':'Continuous', 'BsmtUnfSF':'Continuous', 'TotalBsmtSF':'Continuous', 'Heating':'Nominal', 'HeatingQC':'Ordinal', 'CentralAir':'Nominal', 'Electrical':'Ordinal', '1stFlrSF':'Continuous', '2ndFlrSF':'Continuous', 'LowQualFinSF':'Continuous', 'GrLivArea':'Continuous', 'BsmtFullBath':'Discrete', 'BsmtHalfBath':'Discrete', 'FullBath':'Discrete', 'HalfBath':'Discrete', 'BedroomAbvGr':'Discrete', 'KitchenAbvGr':'Discrete', 'KitchenQual':'Ordinal', 'TotRmsAbvGrd':'Discrete', 'Functional':'Ordinal', 'Fireplaces':'Discrete', 'FireplaceQu':'Ordinal', 'GarageType':'Nominal', 'GarageYrBlt':'Discrete', 'GarageFinish':'Ordinal', 'GarageCars':'Discrete', 'GarageArea':'Continuous', 'GarageQual':'Ordinal', 'GarageCond':'Ordinal', 'PavedDrive':'Ordinal', 'WoodDeckSF':'Continuous', 'OpenPorchSF':'Continuous', 'EnclosedPorch':'Continuous', '3SsnPorch':'Continuous', 'ScreenPorch':'Continuous', 'PoolArea':'Continuous', 'PoolQC':'Ordinal', 'Fence':'Ordinal', 'MiscFeature':'Nominal', 'MiscVal':'Continuous', 'MoSold':'Discrete', 'YrSold':'Discrete', 'SaleType':'Nominal', 'SaleCondition':'Nominal', 'SalePrice':'Continuous'}
featcat = dict([(t,[]) for t in set(backward.values())])
        
for k in backward.keys():
    featcat[backward[k]].append(k)

###############################################################################
# look at features by category
###############################################################################

#==============================================================================
# continuous
#==============================================================================
train_df[featcat['Continuous']].describe()

# -----------------------------------------------------------------------------
# LotFrontage
# -----------------------------------------------------------------------------
# assume missing means zero in this case
train_df.LotFrontage = train_df.LotFrontage.fillna(0)
test_df.LotFrontage = test_df.LotFrontage.fillna(0)
plt.figure(figsize = (16, 9))
plt.hist(train_df.LotFrontage, bins=25)

# the bigger ones are probably corner lots, no?
plt.figure(figsize = (16, 9))
sns.violinplot(x=train_df.LotConfig, y=train_df.LotFrontage)
# hmm

# -----------------------------------------------------------------------------
# LotArea
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df.LotArea, bins=25)
# some big ones - do these correspond to zoning?
plt.figure(figsize = (16, 9))
sns.violinplot(x=train_df.MSZoning, y=train_df.LotArea)
# nope - residential low density

# -----------------------------------------------------------------------------
# MasVnrArea
# -----------------------------------------------------------------------------
# assume missing means zero in this case
train_df.MasVnrArea = train_df.MasVnrArea.fillna(0)
test_df.MasVnrArea = test_df.MasVnrArea.fillna(0)
plt.figure(figsize = (16, 9))
plt.hist(train_df.MasVnrArea, bins=25)

# -----------------------------------------------------------------------------
# BsmtFinSF1
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df.BsmtFinSF1, bins=25)
plt.figure(figsize = (16, 9))
sns.violinplot(x=train_df.BsmtFinType1, y=train_df.BsmtFinSF1)

# -----------------------------------------------------------------------------
# BsmtFinSF2
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df.BsmtFinSF2, bins=25)
plt.figure(figsize = (16, 9))
sns.violinplot(x=train_df.BsmtFinType2, y=train_df.BsmtFinSF2)

# -----------------------------------------------------------------------------
# BsmtUnfSF
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df.BsmtUnfSF, bins=25)

# -----------------------------------------------------------------------------
# TotalBsmtSF
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df.TotalBsmtSF, bins=25)

# -----------------------------------------------------------------------------
# 1stFlrSF
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['1stFlrSF'], bins=25)

# -----------------------------------------------------------------------------
# 2ndFlrSF
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['2ndFlrSF'], bins=25)

# -----------------------------------------------------------------------------
# LowQualFinSF
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['LowQualFinSF'], bins=25)

# -----------------------------------------------------------------------------
# GrLivArea
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['GrLivArea'], bins=25)

# -----------------------------------------------------------------------------
# GarageArea
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['GarageArea'], bins=25)

# -----------------------------------------------------------------------------
# WoodDeckSF
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['WoodDeckSF'], bins=25)

# -----------------------------------------------------------------------------
# OpenPorchSF
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['OpenPorchSF'], bins=25)

# -----------------------------------------------------------------------------
# EnclosedPorch
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['EnclosedPorch'], bins=25)

# -----------------------------------------------------------------------------
# 3SsnPorch
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['3SsnPorch'], bins=25)

# -----------------------------------------------------------------------------
# ScreenPorch
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['ScreenPorch'], bins=25)

# -----------------------------------------------------------------------------
# PoolArea
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['PoolArea'], bins=25)

# -----------------------------------------------------------------------------
# MiscVal
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['MiscVal'], bins=25)
plt.figure(figsize = (16, 9))
sns.violinplot(x=train_df.MiscFeature, y=train_df.MiscVal)

# -----------------------------------------------------------------------------
# SalePrice
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['SalePrice'], bins=25)
plt.figure(figsize = (16, 9))
plt.plot(train_df.GrLivArea, train_df.SalePrice, '.', alpha=0.5)
plt.figure(figsize = (16, 9))
sns.lmplot(x="GrLivArea", y="SalePrice", data=train_df, hue="SaleCondition", truncate=True, size=9)

#==============================================================================
# discrete
#==============================================================================
train_df[featcat['Discrete']].describe()

# -----------------------------------------------------------------------------
# YearBuilt
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['YearBuilt'], bins=25)

# -----------------------------------------------------------------------------
# YearRemodAdd
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
plt.hist(train_df['YearRemodAdd'], bins=25)
plt.figure(figsize = (16, 9))
plt.plot(train_df.YearBuilt, train_df.YearRemodAdd, '.', alpha=0.5)

# -----------------------------------------------------------------------------
# BsmtFullBath
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.BsmtFullBath)

# -----------------------------------------------------------------------------
# BsmtHalfBath
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.BsmtHalfBath)

# -----------------------------------------------------------------------------
# FullBath
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.FullBath)

# -----------------------------------------------------------------------------
# HalfBath
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.HalfBath)

# -----------------------------------------------------------------------------
# BedroomAbvGr
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.BedroomAbvGr)

# -----------------------------------------------------------------------------
# KitchenAbvGr
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.KitchenAbvGr)

# -----------------------------------------------------------------------------
# TotRmsAbvGrd
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.TotRmsAbvGrd)

# -----------------------------------------------------------------------------
# Fireplaces
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.Fireplaces)

# -----------------------------------------------------------------------------
# GarageYrBlt
# -----------------------------------------------------------------------------
# assume null is no garage - test against cars?
plt.figure(figsize = (16, 9))
plt.hist(train_df.GarageCars[train_df.GarageYrBlt.isnull()])
plt.figure(figsize = (16, 9))
sns.violinplot(x=train_df.GarageCars, y=train_df.GarageYrBlt)
# all are zero

# set it to YearBuilt?
train_df.GarageYrBlt = train_df.GarageYrBlt.fillna(train_df.YearBuilt)
test_df.GarageYrBlt = test_df.GarageYrBlt.fillna(test_df.YearBuilt)

plt.figure(figsize = (16, 9))
plt.hist(train_df['GarageYrBlt'], bins=25)
plt.figure(figsize = (16, 9))
plt.plot(train_df.YearBuilt, train_df.GarageYrBlt, '.', alpha=0.5)

# -----------------------------------------------------------------------------
# GarageCars
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.GarageCars)

# -----------------------------------------------------------------------------
# MoSold
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.MoSold)

# -----------------------------------------------------------------------------
# YrSold
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.YrSold)
plt.figure(figsize = (16, 9))
sns.lmplot(x="GrLivArea", y="SalePrice", data=train_df, hue="YrSold", size=9)

# wonder if we should trend these for inflation

#==============================================================================
# ordinal
#==============================================================================
train_df[featcat['Ordinal']].describe()
train_df[featcat['Ordinal']].info()

# -----------------------------------------------------------------------------
# LotShape
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.LotShape)
dct = {'Reg':0, 'IR1':1, 'IR2':2, 'IR3':3}

train_df.LotShape = train_df.LotShape.replace(dct)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.Utilities)
dct = {'AllPub':0, 'NoSeWa':1}

train_df.Utilities = train_df.Utilities.replace(dct)

# -----------------------------------------------------------------------------
# LandSlope
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.LandSlope)
dct = {'Gtl':0, 'Mod':1, 'Sev':2}
train_df.LandSlope = train_df.LandSlope.replace(dct)
test_df.LandSlope = test_df.LandSlope.replace(dct)

# -----------------------------------------------------------------------------
# OverallQual
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.OverallQual)

# -----------------------------------------------------------------------------
# OverallCond
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.OverallCond)

# -----------------------------------------------------------------------------
# ExterQual
# -----------------------------------------------------------------------------
plt.figure(figsize = (16, 9))
sns.countplot(train_df.ExterQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4}
train_df.ExterQual = train_df.ExterQual.replace(dct)
test_df.ExterQual = test_df.ExterQual.replace(dct)
# -----------------------------------------------------------------------------
# ExterCond
# -----------------------------------------------------------------------------
sns.countplot(train_df.ExterCond)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4}
train_df.ExterCond = train_df.ExterCond.replace(dct)
test_df.ExterCond = test_df.ExterCond.replace(dct)

# -----------------------------------------------------------------------------
# BsmtQual
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
test_df.BsmtQual = test_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)
test_df.BsmtQual = test_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# BsmtCond
# -----------------------------------------------------------------------------
train_df.BsmtCond = train_df.BsmtCond.fillna('NA')
test_df.BsmtCond = test_df.BsmtCond.fillna('NA')
sns.countplot(train_df.BsmtCond)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtCond = train_df.BsmtCond.replace(dct)
test_df.BsmtCond = test_df.BsmtCond.replace(dct)

# -----------------------------------------------------------------------------
# BsmtExposure
# -----------------------------------------------------------------------------
train_df.BsmtExposure = train_df.BsmtExposure.fillna('NA')
test_df.BsmtExposure = test_df.BsmtExposure.fillna('NA')
sns.countplot(train_df.BsmtExposure)
dct = {'Gd':0, 'Av':1, 'Mn':2, 'No':3, 'NA':4}
train_df.BsmtExposure = train_df.BsmtExposure.replace(dct)
test_df.BsmtExposure = test_df.BsmtExposure.replace(dct)

# -----------------------------------------------------------------------------
# BsmtFinType1
# -----------------------------------------------------------------------------
train_df.BsmtFinType1 = train_df.BsmtFinType1.fillna('NA')
test_df.BsmtFinType1 = test_df.BsmtFinType1.fillna('NA')
sns.countplot(train_df.BsmtFinType1)
dct = {'GLQ':0, 'ALQ':1, 'BLQ':2, 'Rec':3, 'LwQ':4, 'Unf':5, 'NA':6}
train_df.BsmtFinType1 = train_df.BsmtFinType1.replace(dct)
test_df.BsmtFinType1 = test_df.BsmtFinType1.replace(dct)

# -----------------------------------------------------------------------------
# BsmtFinType2
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
test_df.BsmtQual = test_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)
test_df.BsmtQual = test_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# HeatingQC
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# Electrical
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# KitchenQual
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# Functional
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# FireplaceQu
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# GarageFinish
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# GarageQual
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# GarageCond
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# PavedDrive
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# PoolQC
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

# -----------------------------------------------------------------------------
# Fence
# -----------------------------------------------------------------------------
train_df.BsmtQual = train_df.BsmtQual.fillna('NA')
sns.countplot(train_df.BsmtQual)
dct = {'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'NA':5}
train_df.BsmtQual = train_df.BsmtQual.replace(dct)

