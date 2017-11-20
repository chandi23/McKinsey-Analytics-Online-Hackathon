# McKinsey-Analytics-Online-Hackathon

## Result: Score on **Public Leaderboard** : 	7.947671( **Rank *38* )

## Objective
To predict traffic patterns in each of these four junctions for the next 4 months.

## Approach
We had transaction data of all the customers from Jan 2003 to Dec 2006. The idea is to predict whether the customer will come back in 2007 or not. 

1. The first step was feature extraction from the DateTime variable.
2. Then, feature selection was performed using Boruta package in R.
3. Mean of predictions of 3 xgboost models were taken for final predictions.

## Codes 
#final_model_1.py
Code to create the features from the given input dataset for both validation and final model and applying xgboost.
