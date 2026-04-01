# Churn Prediction: Diaper Size Transition

Predicting customer churn during the diaper size 3 to 4 transition on a direct-to-consumer e-commerce platform, using LightGBM with behavioral and demographic feature engineering.

## Results
- 20% retention improvement among high-risk users
- Validated via A/B test: 2-day to 1-day shipping pilot over 2 months
- Recall-focused model to minimize false negatives (missing at-risk users)

## Problem
Sales dropped sharply as parents transitioned from diaper size 3 to size 4. Standard behavioral features showed no clear signal. The key insight came from connecting delivery zone data to churn patterns: members in same-day delivery zones churned at significantly higher rates, likely switching to a competitor platform offering faster shipping.

## Approach

1. **Data extraction**: Pulled order history, member profiles, child age, login activity, event participation, review ratings, and coupon usage from Snowflake
2. **Feature engineering**:
   - Diaper size purchase flags (Size 1 through 4)
   - Child age in months with target window flag (9 to 18 months = Size 3/4 transition zone)
   - Same-day delivery zone classification based on shipping address
   - Engagement signals: login count, event participation, coupon usage, review rating
   - Premium product purchase history as loyalty signal
3. **Modeling**: LightGBM classifier with Bayesian hyperparameter optimization (BayesSearchCV)
4. **Validation**: A/B test to confirm delivery speed as primary churn driver

## Feature List
| Feature | Description |
|---------|-------------|
| Size1-4PurchaseCount | Purchase count per diaper size |
| ChildAgeMonths | Child age in months |
| TargetAgeGroupYn | Flag for 9-18 month window |
| SameDayDeliveryZoneYn | Whether member is in same-day delivery area |
| LoginCount | Total login count |
| EventCount | Number of events participated |
| ReviewRating | Average review rating |
| CouponYn | Coupon usage flag |
| MemberGrade | Membership tier |
| PremiumPurchaseCount | Premium product line purchase count |

## Tech Stack
- Python, LightGBM, Scikit-learn, Scikit-optimize
- Snowflake, Azure Databricks
- MLflow (experiment tracking)
- Pandas, NumPy

## Note
Data and database credentials have been removed. All table and column names have been anonymized. Model logic and feature engineering are fully preserved.
