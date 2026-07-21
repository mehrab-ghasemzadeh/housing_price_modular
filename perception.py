import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_neighborhood_mean_for_column(df, column, query):
    df_ = df.copy()
    temp = df_[column].median()
    return {
        f"{column}_mean_in_neighborhood" : temp,
        f"{column}_difference_with_neighborhood_mean" : query[column] - temp,
        f"{column}_resemblence_with_neighborhood_mean" : query[column] / temp,
    }

def calc_neighborhood_most_val_for_column(df, column, query_val):
    df_ = df.copy()
    if query_val:
        rows_of_interest = len(df_[df_[column] > 0])
    else:
        rows_of_interest = len(df_[df_[column] == 0])
    rows_of_interest_percentage = (rows_of_interest / len(df_))
    # * 100
    return{
        f"num_{column}_eq_query":rows_of_interest,
        f"percentage_{column}_eq_query":rows_of_interest_percentage
    }

def calc_relavence(query, df_more, df_new):
    df_more = df_more.copy()
    df_new = df_new.copy()
    res = {}
    df_more = df_more[df_more['id_neighbourhood'] == query['id_neighbourhood']]
    df_new = df_new[df_new['id_neighbourhood'] == query['id_neighbourhood']]
    if len(df_more) < 5:
        return {
            'status' : 0,
            'content' : "Not enough data on this neighborhood is present at this moment."
        }
    res["status"] = 1
    res['content'] = {}
    for col in ["area","num_bedrooms","floor"]:
        temp_res = calc_neighborhood_mean_for_column(df_more, col, query)
        for c in temp_res:
            res['content'][c] = temp_res[c]
    for col in ["parking","elevator","storeHouse"]:
        temp_res = calc_neighborhood_most_val_for_column(df_more, col, query[col])
        for c in temp_res:
            res['content'][c] = temp_res[c]
    for key in res['content']:
        res['content'][key] = float(f"{res['content'][key]:.2f}")
    res['content']['mean_price_per_area_of_neighborhood'] = df_new['price_per_area'].median()
    return res
