import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from datetime import datetime

def calc_building_age_threshold(building_age):
    building_age_thresholds = {2:1,5:2,10:3,20:6}
    for key in building_age_thresholds:
        if building_age <= key:
            return building_age_thresholds[key]
    else:
        return 1000

def get_threshold_range(val, base=5, percentage=5):
    divisor = 100/percentage
    return{
        'min':min(val - base, val - (val / divisor)),
        'max':max(val + base, val + (val / divisor)),
    }

def filter_neighbors(df, query):
    df_ = df.copy()
    # ! neighborhood
    df_temp = df_[df_['id_neighbourhood'] == query['id_neighbourhood']]
    if len(df_temp):
        df_ = df_temp
    else:
        return{
            'status':0,
            'content':'There is not enough info on this neighborhood present.'
        }
    # ! area
    for percentage in [5, 10, 15, 20]:
        temp_range = get_threshold_range(query['area'], percentage)
        df_temp = df_[df_['area'] > temp_range['min']]
        df_temp = df_temp[df_temp['area'] < temp_range['max']]
        if len(df_temp):
            df_ = df_temp
            # area_percentage_range = percentage
            break
    # ! building age
    df_temp = df_[df_['building_age'] > query['building_age'] - calc_building_age_threshold(query['building_age'])]
    df_temp = df_temp[df_temp['building_age'] < query['building_age'] + calc_building_age_threshold(query['building_age'])]
    if len(df_temp):
        df_ = df_temp
    # ! parking
    df_temp = df_[df_['parking'] == query['parking']]
    if len(df_temp):
        df_ = df_temp
    elif query['parking']:
        df_temp = df_[df_['parking'] >= 1]
        if len(df_temp):
            df_ = df_temp
    else:
        df_temp = df_[df_['parking'] == 0]
        if len(df_temp):
            df_ = df_temp
    # ! elevator and floor
    if abs(query['floor']) < 3:
        df_temp = df_[df_['floor'] < 3]
        df_temp = df_temp[df_temp['floor'] > -3]
        if len(df_temp):
            df_ = df_temp
    else:
        if query['elevator']:
            df_temp = df_[df_['elevator'] == 1]
        if len(df_temp):
            df_ = df_temp
    # ! num_bedrooms
    temp_range = get_threshold_range(query['num_bedrooms'], base=1, percentage=50)
    df_temp = df_[df_['num_bedrooms'] > temp_range['min']]
    df_temp = df_temp[df_temp['num_bedrooms'] < temp_range['max']]
    if len(df_temp):
        df_ = df_temp
    # ! 'balcony'
    df_temp = df_[df_['balcony'] == query['balcony']]
    if len(df_temp):
        df_ = df_temp
    # ! 'is_luxury'
    df_temp = df_[df_['is_luxury'] == query['is_luxury']]
    if len(df_temp):
        df_ = df_temp
    # ! 'is_modern'
    df_temp = df_[df_['is_modern'] == query['is_modern']]
    if len(df_temp):
        df_ = df_temp
    # ! 'janitor'
    df_temp = df_[df_['janitor'] == query['janitor']]
    if len(df_temp):
        df_ = df_temp
    # ! 'master_room'
    df_temp = df_[df_['master_room'] == query['master_room']]
    if len(df_temp):
        df_ = df_temp
    # ! 'pool'
    df_temp = df_[df_['pool'] == query['pool']]
    if len(df_temp):
        df_ = df_temp
    # ! 'security'
    df_temp = df_[df_['security'] == query['security']]
    if len(df_temp):
        df_ = df_temp
    # ! 'gym'
    df_temp = df_[df_['gym'] == query['gym']]
    if len(df_temp):
        df_ = df_temp
    
    return {
        'status':1,
        'content':df_
    }

def calculate_cosine_similarity_scores(query, neighbors):
    neighbors = neighbors.copy()
    WANTED_COLUMNS = ['building_age', 'num_bedrooms', 'area', 'id_neighbourhood', 'floor', 'parking', 'elevator', 'storeHouse', 'date_publish']
    if neighbors.empty:
        return pd.Series(dtype=float)
    neighbors = neighbors[WANTED_COLUMNS].copy()
    try:
        neighbors['date_parsed'] = pd.to_datetime(neighbors['date_publish'], format="%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format in date_publish: {e}")
    feature_cols = [col for col in WANTED_COLUMNS if col != 'date_publish']
    query_df = pd.DataFrame([query])[feature_cols].fillna(0)
    combined = pd.concat([query_df, neighbors[feature_cols].fillna(0)], ignore_index=True)
    if len(neighbors) > 1:
        combined = (combined - combined.min()) / (combined.max() - combined.min())
    combined = combined.fillna(0)
    query_normalized = combined.iloc[[0]].values
    neighbors_normalized = combined.iloc[1:].values
    similarities = cosine_similarity(query_normalized, neighbors_normalized)[0]
    days_old = (datetime.today() - neighbors['date_parsed']).dt.days
    recency_boost = (days_old <= 30).astype(int) + 1
    similarities = similarities * recency_boost.values
    return pd.Series(similarities, index=neighbors.index, name="score")

def calculate_similarity_scores(query, neighbors):
    WANTED_COLUMNS = ['building_age', 'num_bedrooms', 'area', 'floor', 'parking', 'elevator', 'storeHouse', 'date_publish']
    if neighbors.empty:
        return pd.Series(dtype=float)
    neighbors = neighbors[WANTED_COLUMNS].copy()
    try:
        neighbors['date_parsed'] = pd.to_datetime(neighbors['date_publish'], format="%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format in date_publish: {e}")
    feature_cols = [col for col in WANTED_COLUMNS if col != 'date_publish']
    query_values = [query[col] for col in feature_cols]
    neighbor_values = neighbors[feature_cols].values
    floor_distances = np.abs(neighbor_values[:, feature_cols.index('floor')] - query_values[feature_cols.index('floor')])
    floor_distances = np.where(neighbors['elevator'] == 1, floor_distances / 3, floor_distances)
    distances = np.abs(neighbor_values - query_values).sum(axis=1)
    distances = distances - np.abs(neighbor_values[:, feature_cols.index('floor')] - query_values[feature_cols.index('floor')]) + floor_distances
    similarities = 100 - distances
    days_old = (datetime.today() - neighbors['date_parsed']).dt.days
    weight = np.ones(len(neighbors))
    weight[(similarities > 95) & (days_old < 45)] = 1.45
    weight[(similarities > 95) & (days_old < 30)] = 1.70
    weight[(similarities > 95) & (days_old < 15)] = 2.00
    weight[(similarities > 90) & (days_old < 30)] = 1.50
    weight[(similarities > 90) & (days_old < 15)] = 1.65
    weight[similarities < 80] = 0.80
    neighbors['score'] = similarities
    neighbors['weight'] = similarities * weight
    return neighbors[['score', 'weight']]

def weighted_sum(df):
    return (df['price_per_area'] * df['weight']).sum() / df['weight'].sum()

def calculate_effecctive_price(df, query):
    df = df.copy()
    effective_prices = np.array([])
    def get_age_relavance(a, b=query['building_age']):
        age_scores = {1:0, 3:0.01, 7:0.025, 15:0.04, 200000:0.08}
        for key in age_scores:
            if abs(a-b) < key:
                return age_scores[key]
    for index, row in df.iterrows():
        row = row.to_dict()
        price = row['price_per_area']
        if query['building_age'] < row['building_age']:
            price *= (1 + get_age_relavance(price))
        else:
            price *= (1 - get_age_relavance(price))
        if row['num_bedrooms'] - query['num_bedrooms'] > 2:
            price *= 0.98
        elif query['num_bedrooms'] - row['num_bedrooms'] > 2:
            price *= 0.98
        
        if query['elevator'] == row['elevator'] and query['elevator']:
            if query['floor'] > 5:
                if query['floor'] - row['floor'] >= 2:
                    price *= 1.01
                elif row['floor'] - query['floor'] >= 2:
                    price *= 0.99
        elif query['elevator'] == row['elevator'] and not query['elevator']:
            if query['floor'] >= 2:
                if query['floor'] - row['floor'] >= 2:
                    price *= (1 - ((query['floor'] - row['floor']) * 0.003))
                elif row['floor'] - query['floor'] >= 2:
                    price *= (1 - (abs(query['floor'] - row['floor']) * 0.003))
        elif not query['elevator'] and row['elevator']:
            if query['floor'] > row['floor']:
                price *= (1 - ((query['floor'] - row['floor']) * 0.003))
            elif row['floor'] > query['floor']:
                price *= (1 - (abs(query['floor'] - row['floor']) * 0.003))
        elif not row['elevator'] and query['elevator']:
            if query['floor'] < row['floor']:
                price *= (1 - ((query['floor'] - row['floor']) * 0.003))
            elif row['floor'] < query['floor']:
                price *= (1 - (abs(query['floor'] - row['floor']) * 0.003))

        if query['parking'] > row['parking']:
            price *= 1.02
        elif query['parking'] < row['parking']:
            price *= 0.98
        if query['storeHouse'] > row['storeHouse']:
            price *= 1.005
        elif query['storeHouse'] < row['storeHouse']:
            price *= 0.995
        # ! optional features
        print(price, end='\t')
        if query['balcony'] > row['balcony']:
            price *= 1.01
        elif row['balcony'] > query['balcony']:
            price *= 0.99
        if query['is_luxury'] > row['is_luxury']:
            price *= 1.02
        elif row['is_luxury'] > query['is_luxury']:
            price *= 0.98
        if query['is_modern'] > row['is_modern']:
            price *= 1.015
        elif row['is_modern'] > query['is_modern']:
            price *= 0.985
        if query['janitor'] > row['janitor']:
            price *= 1.02
        elif row['janitor'] > query['janitor']:
            price *= 0.98
        if query['master_room'] > row['master_room']:
            price *= 1.005
        elif row['master_room'] > query['master_room']:
            price *= 0.995
        if query['pool'] > row['pool']:
            price *= 1.03
        elif row['pool'] > query['pool']:
            price *= 0.97
        if query['security'] > row['security']:
            price *= 1.03
        elif row['security'] > query['security']:
            price *= 0.97
        if query['gym'] > row['gym']:
            price *= 1.025
        elif row['gym'] > query['gym']:
            price *= 0.975
        print(price, end='\n')

        effective_prices = np.append(effective_prices, price)
    return effective_prices











