def regressor_predict(data: dict):
    import joblib
    import pandas as pd
    import os
    
    BASE_MODEL_DIR = "model_reg"
    best_model = joblib.load(os.path.join(BASE_MODEL_DIR, 'best_model.pkl'))
    scaler_X = joblib.load(os.path.join(BASE_MODEL_DIR, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(BASE_MODEL_DIR, 'scaler_y.pkl'))
    
    # Convert dictionary to DataFrame with correct column order
    # cols = ['building_age', 'num_bedrooms', 'area', 'id_neighbourhood', 'floor', 'parking', 'elevator', 'storeHouse']
    cols = ['building_age','num_bedrooms','area','id_neighbourhood','floor','parking','elevator','storeHouse','balcony','is_luxury','is_modern','janitor','master_room','pool','security','gym']
    data_df = pd.DataFrame([data])[cols]  # Ensure columns are in the same order as training
    
    # Use them for predictions on new data
    new_data_scaled = scaler_X.transform(data_df)
    predictions_scaled = best_model.predict(new_data_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    return predictions[0]


# if __name__ == "__main__":
#     new_data = {
#         'area': 130,
#         'id_neighbourhood': 200,
#         'building_age': 20,
#         'floor': 1,
#         'num_bedrooms': 3,
#         'elevator': 1,
#         'parking': 1,
#         'storeHouse': 1,
#     }
#     result = regressor_predict(new_data)
#     print(f"Predicted price per area: {result:.2f}")
