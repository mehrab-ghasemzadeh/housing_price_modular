import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np
from regression import regressor_predict
from filter import (
    filter_neighbors, 
    weighted_sum,
    calculate_cosine_similarity_scores,
    calculate_similarity_scores,
    calculate_effecctive_price
)
from perception import calc_relavence

df = pd.DataFrame({})

def generate_price_trend_plot(df, nid):
    df = df.copy()
    df = df[df['price_per_area'] < df['price_per_area'].quantile(0.99)]
    df = df[df['price_per_area'] > df['price_per_area'].quantile(0.01)]
    df_filtered = df[df['id_neighbourhood'] == nid]
    df = df_filtered.copy()
    df['date_publish'] = pd.to_datetime(df['date_publish'], format='%Y-%m-%d')
    df['year'] = df['date_publish'].dt.year.astype(int)
    df['month'] = df['date_publish'].dt.month.astype(int)
    grouped = df.groupby(['year', 'month']).agg({
        'price_per_area': 'mean',
        'date_publish': 'count'
    }).rename(columns={'date_publish': 'count'})
    grouped.reset_index(inplace=True)
    grouped['year'] = grouped['year'].astype(int)
    grouped['month'] = grouped['month'].astype(int)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(range(len(grouped)), grouped['price_per_area'], marker='o', linewidth=2, color='#667eea')
    ax1.set_ylabel('میانگین قیمت هر متر (میلیارد تومان)', fontsize=12)
    ax1.set_title('میانگین قیمت هر متر املاک آگهی شده در هر ماه', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(grouped['price_per_area']):
        ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    bars = ax2.bar(range(len(grouped)), grouped['count'], alpha=0.7, color='#764ba2')
    ax2.set_ylabel('تعداد آگهی', fontsize=12)
    ax2.set_title('تعداد آگهی های پست شده این منطقه در هر ماه', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(grouped['count']):
        ax2.text(i, v + 0.5, str(int(v)), ha='center', va='bottom', fontsize=8)
    labels = [f"{int(row['year'])}-{int(row['month']):02d}" for _, row in grouped.iterrows()]
    ax2.set_xticks(range(len(grouped)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return image_base64

def predict_new(query={}, data_2y=df, data=df, neighborhoods=df):
    print(query)
    WANTED_COLUMNS = ['date_publish','price_per_area','building_age','num_bedrooms','area','id_neighbourhood','floor','parking','elevator','storeHouse','balcony','is_luxury','is_modern','janitor','master_room','pool','security','gym']
    data = data[WANTED_COLUMNS]
    result = {}

    result_reg = regressor_predict(query)
    result['model_base_regression'] = result_reg

    query_neighbors = filter_neighbors(data[WANTED_COLUMNS], query)
    if query_neighbors['status']:
        score = calculate_similarity_scores(query, query_neighbors['content'])
        query_neighbors_with_score = pd.concat([query_neighbors['content'], score], axis=1)
        result_filter = weighted_sum(query_neighbors_with_score)
        result['instance_base_regression'] = result_filter
        result['instance_base_neighbors'] = query_neighbors_with_score.to_dict(orient='records')
        
        trend_plot = generate_price_trend_plot(data_2y, query['id_neighbourhood'])
        result['trend_plot'] = trend_plot
        
        avg_weight = query_neighbors_with_score['weight'].mean()
        if result_filter and result_reg:
            if len(query_neighbors_with_score) == 1:
                if avg_weight > 90:
                    result_filter_weight = 9
                    result_reg_weight = 2
                else:
                    result_filter_weight = 8
                    result_reg_weight = 4
            elif len(query_neighbors_with_score) <= 3:
                if avg_weight > 80:
                    result_filter_weight = 8
                    result_reg_weight = 3
                else:
                    result_filter_weight = 9
                    result_reg_weight = 5
            else:
                if avg_weight > 70:
                    result_filter_weight = 10
                    result_reg_weight = 1
                else:
                    result_filter_weight = 8
                    result_reg_weight = 2

        query_neighbors_with_score['price_per_area'] = calculate_effecctive_price(query_neighbors_with_score, query)

        final_prediction = ((result_filter * result_filter_weight) + (result_reg * result_reg_weight)) / (result_filter_weight + result_reg_weight)
        result['final_prediction'] = final_prediction
        result['min_value'] = ((query_neighbors_with_score['price_per_area'].min() * result_filter_weight) + (result_reg * result_reg_weight)) / (result_filter_weight + result_reg_weight)
        result['max_value'] = ((query_neighbors_with_score['price_per_area'].max() * result_filter_weight) + (result_reg * result_reg_weight)) / (result_filter_weight + result_reg_weight)

        res = calc_relavence(query, data_2y, data)
        neighborhood_monotony = 1
        if res['status']:
            monotony_weights = np.array([0.9, 0.4, 0.05, 0.8, 0.4, 0.5])
            neighborhood_monotony = (
                    (res['content']['area_resemblence_with_neighborhood_mean'] * monotony_weights[0]) +
                    (res['content']['num_bedrooms_resemblence_with_neighborhood_mean'] * monotony_weights[0]) +
                    (res['content']['floor_resemblence_with_neighborhood_mean'] * monotony_weights[0]) +
                    (res['content']['percentage_parking_eq_query'] * monotony_weights[0]) +
                    (res['content']['percentage_elevator_eq_query'] * monotony_weights[0]) +
                    (res['content']['percentage_storeHouse_eq_query'] * monotony_weights[0])
                ) / monotony_weights.sum()
        result['neighborhood_ifno'] = res['content']
        result['neighborhood_monotony'] = neighborhood_monotony
        result['total_price'] = final_prediction * query['area']
        
        return {
            'status': 1,
            'content': result
        }
    else:
        return {
            'status': 0,
            'content': {
                'error': query_neighbors['content'],
                'model_base_regression': result_reg
            }
        }

# if __name__ == "__main__":
#     data_2y = pd.read_csv('data/data_final_2y.csv')
#     data = pd.read_csv('data/data_final.csv', index_col=0)
#     data = data[data['price_per_area'] > data['price_per_area'].quantile(0.01)]
#     neighborhoods = pd.read_csv('data/neighborhood_ids.csv')

#     query = {
#         'area': 135,
#         'id_neighbourhood': 200,
#         'building_age': 18,
#         'floor': 1,
#         'num_bedrooms': 1,
#         'elevator': 1,
#         'parking': 1,
#         'storeHouse': 1,
#     }

#     r = predict_new(query=query, data=data, data_2y=data_2y, neighborhoods=neighborhoods)
#     # for k in r:
#     #     print(k, r[k], sep='\n', end='\n\n')