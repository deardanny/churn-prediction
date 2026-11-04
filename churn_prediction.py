# Databricks notebook source
# MAGIC %pip install scikit-optimize scikit-learn lightgbm snowflake-connector-python snowflake-sqlalchemy mlflow hyperopt

# COMMAND ----------

import os
import re
import pandas as pd
import numpy as np
import mlflow
import torch, gc

from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier, plot_importance
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# COMMAND ----------

# ── DATABASE CONNECTION ──────────────────────────────────────────────────────
# Credentials must be set as environment variables before running:
#   SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT
#   SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, SNOWFLAKE_WAREHOUSE

import snowflake.connector

current_date = datetime.now()

con = snowflake.connector.connect(
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"]
)
cur = con.cursor()

# COMMAND ----------

# ── DATA EXTRACTION ──────────────────────────────────────────────────────────

# Product information
cur.execute("SELECT * FROM PRODUCT_TABLE")
data = cur.fetchall()
product_columns = [
    'ProductNo', 'SiteName', 'ProductName', 'Class1', 'Class2', 'Class3',
    'ProviderNo', 'BrandNo', 'BrandName', 'ProductType', 'ModelName', 'SellYN',
    'Maker', 'DeliveryProviderNo', 'add_info1', 'add_info2', 'add_info3',
    'add_info4', 'add_info5', 'add_info6', 'add_info7', 'add_info8', 'add_info9',
    'add_info10', 'StockUnit', 'CreateDts', 'UpdateDts', 'IFDate',
    'LikesCount', 'LocalViewCount', 'HiddenDealProductYN', 'DisplayYn', 'sf_date', 'stime'
]
product_df = pd.DataFrame(data, columns=product_columns)
product_df = product_df[['ProductNo', 'BrandName', 'add_info7', 'add_info8', 'add_info9']]

# Order item information (filtered to recent orders from 2024 onwards)
cur.execute("SELECT OrderNo, OrderItemNo, SiteName, ProductNo, ProductName, SellPrice, ProductPrice, ProductCount FROM ORDER_ITEM_TABLE")
data = cur.fetchall()
orderitem_columns = ['OrderNo', 'OrderItemNo', 'SiteName', 'ProductNo', 'ProductName', 'SellPrice', 'ProductPrice', 'ProductCount']
orderitem_df = pd.DataFrame(data, columns=orderitem_columns)
orderitem_df['OrderDate'] = pd.to_datetime(orderitem_df['OrderNo'].str[:8], format='%Y%m%d')
orderitem_df_recent = orderitem_df[orderitem_df['OrderDate'] > pd.to_datetime('2024-01-01')]

# Order member information
cur.execute("SELECT OrderNo, MemberID, MbrNo, OrderAmt, PayAmt, DeliveryPrice, ReserverAmt, OrderChannel, PremiumOrderYn, OrderMemberGrade FROM ORDER_TABLE")
data = cur.fetchall()
order_columns = ['OrderNo', 'MemberID', 'MbrNo', 'OrderAmt', 'PayAmt', 'DeliveryPrice', 'ReserverAmt', 'OrderChannel', 'PremiumOrderYn', 'OrderMemberGrade']
ordermember_df = pd.DataFrame(data, columns=order_columns)

# Merge order and product data
order_df = pd.merge(orderitem_df_recent, ordermember_df, on='OrderNo', how='left')
order_df = pd.merge(order_df, product_df, on='ProductNo', how='left')

# COMMAND ----------

# ── FEATURE ENGINEERING: DIAPER SIZE TRANSITION ───────────────────────────────
# Core hypothesis: members who have purchased Size 3 diapers but not Size 4
# are at highest churn risk during the transition window.

def add_size_purchase_flag(df, step):
    col_name = f'Size{step}Purchased'
    df[col_name] = df['ProductName'].apply(
        lambda x: 1 if re.search(rf'(?<!\d){step}step', str(x)) else 0
    )

for step in range(1, 5):
    add_size_purchase_flag(order_df, step)

# Flag premium product line purchases (used as engagement signal)
PREMIUM_PRODUCT_KEYWORD = "PremiumLine"
BABY_ACCESSORY_KEYWORD = "BabyAccessory"

order_df['PremiumPurchaseYn'] = order_df['ProductName'].apply(
    lambda x: 1 if PREMIUM_PRODUCT_KEYWORD in str(x) else 0
)
order_df['BabyAccessoryPurchaseYn'] = order_df['ProductName'].apply(
    lambda x: 1 if BABY_ACCESSORY_KEYWORD in str(x) else 0
)

# Aggregate to member level
order_df = order_df.groupby('MemberID').agg({
    'OrderNo': 'count',
    'SellPrice': 'sum',
    'Size1Purchased': 'sum',
    'Size2Purchased': 'sum',
    'Size3Purchased': 'sum',
    'Size4Purchased': 'sum',
    'OrderDate': 'max',
    'OrderMemberGrade': 'last',
    'PremiumPurchaseYn': 'sum',
    'BabyAccessoryPurchaseYn': 'sum'
}).reset_index()

order_df.columns = [
    'MemberID', 'TotalPurchaseCount', 'TotalPurchaseAmount',
    'Size1PurchaseCount', 'Size2PurchaseCount', 'Size3PurchaseCount', 'Size4PurchaseCount',
    'LastPurchaseDate', 'MemberGrade', 'PremiumPurchaseCount', 'BabyAccessoryPurchaseCount'
]

# COMMAND ----------

# ── MEMBER DATA ───────────────────────────────────────────────────────────────

# Member base info
cur.execute("SELECT MemberNo, MemberID, MemberJoinSite, Gender, RecommenderId, DormancyYn, CreateDts FROM MEMBER_TABLE")
data = cur.fetchall()
member_columns = ['MemberNo', 'MemberID', 'MemberJoinSite', 'Gender', 'RecommenderId', 'DormancyYn', 'MemberJoinDts']
member_df = pd.DataFrame(data, columns=member_columns)
member_df = member_df.groupby('MemberID').agg({
    'Gender': 'last',
    'DormancyYn': 'last',
    'MemberJoinDts': 'max'
}).reset_index()

# Child information with age in months
cur.execute("SELECT MemberID, ChildrenNo, Birthday, Gender FROM MEMBER_CHILDREN_TABLE")
data = cur.fetchall()
memberchild_df = pd.DataFrame(data, columns=['MemberID', 'ChildrenNo', 'ChildBirthday', 'ChildGender'])
memberchild_df['ChildBirthday'] = pd.to_datetime(memberchild_df['ChildBirthday'])
memberchild_df['ChildAgeMonths'] = memberchild_df['ChildBirthday'].apply(
    lambda x: (current_date.year - x.year) * 12 + (current_date.month - x.month)
)
# Target age window: 9-18 months corresponds to the Size 3 to 4 transition
memberchild_df['TargetAgeGroupYn'] = memberchild_df['ChildAgeMonths'].apply(
    lambda x: 1 if 9 <= x <= 18 else 0
)
memberchild_df = memberchild_df.groupby('MemberID').agg({
    'ChildrenNo': 'max',
    'ChildGender': 'last',
    'ChildAgeMonths': 'max',
    'TargetAgeGroupYn': 'max'
}).reset_index()

# Login activity
cur.execute("SELECT MemberID, CreateDts FROM LOGIN_LOG_TABLE")
data = cur.fetchall()
loginlog_df = pd.DataFrame(data, columns=['MemberID', 'LoginLog'])
loginlog_df = loginlog_df.groupby('MemberID').agg({'LoginLog': 'count'}).reset_index()
loginlog_df.columns = ['MemberID', 'LoginCount']

# Event participation
cur.execute("SELECT MemberID, CreateDts, EventNo FROM EVENT_TABLE")
data = cur.fetchall()
event_df = pd.DataFrame(data, columns=['MemberID', 'EvtApplyDts', 'EventNo'])
event_df = event_df.groupby('MemberID').agg({'EvtApplyDts': ['max', 'count']}).reset_index()
event_df.columns = ['MemberID', 'LastEventApply', 'EventCount']

# Review ratings
cur.execute("SELECT MemberID, Point FROM REVIEW_TABLE")
data = cur.fetchall()
review_df = pd.DataFrame(data, columns=['MemberID', 'ReviewRating'])
review_df = review_df.groupby('MemberID').agg({'ReviewRating': 'mean'}).reset_index()

# Coupon usage
cur.execute("SELECT MemberID, UseYn FROM COUPON_TABLE")
data = cur.fetchall()
coupon_df = pd.DataFrame(data, columns=['MemberID', 'CouponYn'])
coupon_df = coupon_df.groupby('MemberID').agg({'CouponYn': 'max'}).reset_index()

# COMMAND ----------

# ── ROCKET DELIVERY ZONE FEATURE ─────────────────────────────────────────────
# Key insight: members in same-day delivery zones showed significantly higher
# churn rates, likely switching to competitor platforms offering faster shipping.
# Delivery zones are classified as same-day vs standard based on region codes.

SAME_DAY_DELIVERY_REGIONS = [
    'Seoul',
    'Gyeonggi-Goyang', 'Gyeonggi-Gwacheon', 'Gyeonggi-Gwangmyeong',
    'Gyeonggi-Guri', 'Gyeonggi-Gunpo', 'Gyeonggi-Gimpo',
    'Gyeonggi-Bucheon', 'Gyeonggi-Seongnam', 'Gyeonggi-Suwon',
    'Gyeonggi-Siheung', 'Gyeonggi-Ansan', 'Gyeonggi-Anyang',
    'Gyeonggi-Yangju', 'Gyeonggi-Osan', 'Gyeonggi-Yongin',
    'Gyeonggi-Uiwang', 'Gyeonggi-Uijeongbu', 'Gyeonggi-Paju',
    'Gyeonggi-Pocheon', 'Gyeonggi-Hanam',
    'Incheon', 'Sejong', 'Daejeon', 'Daegu', 'Gwangju', 'Ulsan', 'Busan',
    'Chungbuk-Cheongju', 'Chungnam-Asan', 'Chungnam-Cheonan',
    'Jeonbuk-Jeonju',
    'Gyeongbuk-Gyeongsan', 'Gyeongbuk-Gumi', 'Gyeongbuk-Gimcheon',
    'Gyeongnam-Gimhae', 'Gyeongnam-Yangsan', 'Gyeongnam-Changwon'
]

STANDARD_DELIVERY_REGIONS = [
    'Gyeonggi-Gapyeong', 'Gyeonggi-Gwangju', 'Gyeonggi-Anseong',
    'Gyeonggi-Yangpyeong', 'Gyeonggi-Yeoju', 'Gyeonggi-Yeoncheon',
    'Gyeonggi-Icheon', 'Gyeonggi-Pyeongtaek', 'Gyeonggi-Hwaseong',
    'Incheon-Ganghwa', 'Gangwon',
    'Chungbuk-Goesan', 'Chungbuk-Danyang', 'Chungbuk-Boeun',
    'Chungbuk-Yeongdong', 'Chungbuk-Okcheon', 'Chungbuk-Eumseong',
    'Chungbuk-Jecheon', 'Chungbuk-Jincheon', 'Chungbuk-Chungju',
    'Chungnam-Gongju', 'Chungnam-Geumsan', 'Chungnam-Nonsan',
    'Chungnam-Dangjin', 'Chungnam-Boryeong', 'Chungnam-Buyeo',
    'Chungnam-Seosan', 'Chungnam-Seocheon', 'Chungnam-Yesan',
    'Jeonbuk-Gochang', 'Jeonbuk-Gunsan', 'Jeonbuk-Gimje',
    'Jeonbuk-Namwon', 'Jeonbuk-Muju', 'Jeonbuk-Buan',
    'Jeonbuk-Wanju', 'Jeonbuk-Iksan', 'Jeonbuk-Jeongeup',
    'Jeonnam',
    'Gyeongbuk-Gyeongju', 'Gyeongbuk-Andong', 'Gyeongbuk-Pohang',
    'Gyeongnam-Geoje', 'Gyeongnam-Jinju', 'Gyeongnam-Tongyeong',
    'Jeju'
]

standard_region_pattern = '|'.join(STANDARD_DELIVERY_REGIONS)

# COMMAND ----------

# ── DATASET MERGE ─────────────────────────────────────────────────────────────

final_df = pd.merge(member_df, order_df, on='MemberID', how='inner')
final_df = pd.merge(final_df, memberchild_df, on='MemberID', how='inner')
final_df = pd.merge(final_df, loginlog_df, on='MemberID', how='inner')
final_df = pd.merge(final_df, event_df, on='MemberID', how='left')
final_df = pd.merge(final_df, review_df, on='MemberID', how='left')
final_df = pd.merge(final_df, coupon_df, on='MemberID', how='left')

# Fill missing values for engagement features
final_df[['EventCount', 'ReviewRating']] = final_df[['EventCount', 'ReviewRating']].fillna(0)
final_df[['CouponYn']] = final_df[['CouponYn']].fillna('N')

# COMMAND ----------

# ── TARGET VARIABLE ───────────────────────────────────────────────────────────
# Target: whether a member purchased Size 4 diapers (1 = purchased, 0 = churned)
# Scope: only members who have purchased Size 3 (active at-risk segment)

final_df['TargetYN'] = final_df['Size4PurchaseCount'].apply(lambda x: 1 if x > 0 else 0)
final_df = final_df[final_df['Size3PurchaseCount'] > 0]

# COMMAND ----------

# ── ENCODING ─────────────────────────────────────────────────────────────────

encoders = {}
categorical_cols = ['Gender', 'DormancyYn', 'MemberGrade', 'ChildGender', 'CouponYn']

for col in categorical_cols:
    enc = LabelEncoder()
    final_df.loc[:, col] = enc.fit_transform(final_df[col])
    encoders[col] = enc

final_df.loc[:, 'MemberJoinDts'] = pd.to_numeric(final_df['MemberJoinDts'])
final_df.loc[:, 'LastPurchaseDate'] = pd.to_numeric(final_df['LastPurchaseDate'])
final_df.loc[:, 'LastEventApply'] = pd.to_numeric(final_df['LastEventApply'])

# COMMAND ----------

# ── MODELING ─────────────────────────────────────────────────────────────────

X = final_df.drop(columns=['MemberID', 'TargetYN', 'Size4PurchaseCount'])
y = final_df['TargetYN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train with best parameters found via Bayesian optimization (see tuning section below)
model = LGBMClassifier(
    random_state=0,
    bagging_fraction=0.8923,
    feature_fraction=0.9,
    max_depth=0,
    min_data_in_leaf=25,
    num_leaves=46
)
model.fit(X_train, y_train)

# COMMAND ----------

# ── EVALUATION ────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
print(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred, average="weighted"):.4f}')
print(f'Recall:    {recall_score(y_test, y_pred, average="weighted"):.4f}')
print(f'F1 Score:  {f1_score(y_test, y_pred, average="weighted"):.4f}')

# COMMAND ----------

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────

import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# ── HYPERPARAMETER TUNING (BAYESIAN OPTIMIZATION) ─────────────────────────────
# Run this section to find optimal hyperparameters.
# Results were used to set the parameters in the modeling section above.

param_grid = {
    'num_leaves': Integer(7, 63),
    'min_data_in_leaf': Integer(1, 30),
    'max_depth': Integer(-1, 7),
    'bagging_fraction': Real(0.6, 0.9),
    'feature_fraction': Real(0.6, 0.9)
}

bayes_search = BayesSearchCV(LGBMClassifier(random_state=0), param_grid, cv=5)
bayes_search.fit(X_train, y_train)

best_params = bayes_search.best_params_
print('Best parameters:', best_params)

tuned_model = LGBMClassifier(**best_params)
tuned_model.fit(X_train, y_train)

y_pred_tuned = tuned_model.predict(X_test)
print('Accuracy after tuning:', accuracy_score(y_test, y_pred_tuned))

# COMMAND ----------

# ── PREDICTION ON FULL DATASET ────────────────────────────────────────────────
# Generate churn predictions for all members in the dataset

X_test_with_id = final_df.loc[X_test.index, ['MemberID']]
results_df = pd.DataFrame({
    'MemberID': X_test_with_id['MemberID'],
    'Prediction': model.predict(X_test)
})

results_df = pd.merge(
    results_df,
    final_df[['MemberID', 'Size1PurchaseCount', 'Size2PurchaseCount',
              'Size3PurchaseCount', 'Size4PurchaseCount', 'TargetYN']],
    on='MemberID',
    how='left'
)

print(results_df['TargetYN'].value_counts())
