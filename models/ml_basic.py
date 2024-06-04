import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from helpers import get_sample
import joblib

cool_Ta = 23
cool_RH = 50

# Import dataset and trim to only required columns
def import_data(features, output):
    # Import dataset
    df = pd.read_csv('./dataset/230322_OlderPredictTc_data_thermal.csv')

    # Only use previous values from same individual
    df['previous_tre_int'] = df.groupby('id_all')['tre_int'].shift(1)
    df['previous_mtsk_int'] = df.groupby('id_all')['mtsk_int'].shift(1)

    # Select only time > 0
    df = df[df.time > 0]

    # IMPORTANT: REPLACE 9% RH during cooling in heatwave 3 with the correct value of 50%
    selector = (df['study'] == 'heatwave 3 (cooling)') & (df['ta_set'] == 23)
    df.loc[selector, 'rh_set'] = 50

    # Unique ID to identify an individual under a certain condition
    df['unique_id'] = df['study'].astype(str) + '_' + df['condition'].astype(str) + '_' + df['id_all'].astype(str)

    # Select only features and output
    df = df[features + output + ['id_all', 'study', 'condition', 'time', 'unique_id']]

    # Create train_df based on participants assigned to training set
    train_ids = [46, 34, 68, 30, 40, 98, 89, 65, 24, 58, 85, 67, 28, 39, 35, 77, 26,
                 80, 70, 37, 52, 56, 74, 78, 71, 60, 86, 43, 91, 82, 22, 59, 21, 87,
                 95, 66, 44, 25, 76, 94, 53, 32, 73, 23, 49]
    train_df = df[df['id_all'].isin(train_ids)]

    # Reset index
    train_df.reset_index(inplace=True)

    return train_df

# Scale the dataset between 0 to 1
def scale_data(train_df, features, output):
    features_scaler = MinMaxScaler(feature_range=(0,1))
    output_scaler = MinMaxScaler(feature_range=(0,1))

    # Fit scalers
    train_features = features_scaler.fit_transform(train_df[features])
    train_output = output_scaler.fit_transform(train_df[output])

    return features_scaler, output_scaler, train_features, train_output

# Train the model
def train_model(model, train_features, train_output):
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    model.fit(train_features, train_output)

    return model

# SIMULATE
# Simulate initial 60 minutes
def simulate_initial(body_parameters, features_scaler, output_scaler, model):
    # Initial ambient conditions for cooling environment (23°C 50% RH)
    initial_ambient = [cool_Ta, cool_RH]
    # Initial core and skin temp
    body_conditions = [37, 32]
    # Simulate for 60 mins
    for i in range(60):
        X = np.array(body_parameters + initial_ambient + body_conditions).reshape(1, -1)
        X = features_scaler.transform(X)
        y = model.predict(X)
        y = output_scaler.inverse_transform(y)
        body_conditions = y[0].tolist()
    # Return core and skin temp at end of 60 mins (e.g. [37, 32])
    return body_conditions

# Run and save output results for a model
def run_and_save_trial(study, condition, features, features_scaler, output_scaler, model, model_name):
    # Get sample
    sample = get_sample(study, condition)
    # Group by unique id
    grouped = sample.groupby('id_all', sort=False)
    all_predicted_values = []
    for name, group in grouped:
        predicted_values = False
        for index, row in group.iterrows():
            # Get initial conditions
            if not predicted_values:
                # Body parameters taken from data
                body_parameters = row[features[:-4]].tolist()
                predicted_values = [simulate_initial(body_parameters, features_scaler, output_scaler, model)]
                continue
            # Calculate next temp values
            X = np.array(row[features[:-2]].tolist() + predicted_values[-1]).reshape(1, -1)
            X = features_scaler.transform(X)
            y = model.predict(X)
            y = output_scaler.inverse_transform(y)
            predicted_values.append(y[0].tolist())
        all_predicted_values += predicted_values

    all_core_temps = np.array(all_predicted_values)[:, 0]
    all_skin_temps = np.array(all_predicted_values)[:, 1]
    # Get predicted columns
    df = pd.DataFrame(all_core_temps, columns=["tre_predicted"])
    df["mtsk_predicted"] = all_skin_temps
    # Save to csv
    df.to_csv('results/{}-{}-{}.csv'.format(model_name, study, condition), index=False)
    # Calculate RMSE
    tre_rmse = np.sqrt(mean_squared_error(sample['tre_int'], df['tre_predicted']))
    mtsk_rmse = np.sqrt(mean_squared_error(sample['mtsk_int'], df['mtsk_predicted']))
    return tre_rmse, mtsk_rmse

def train_and_run_all(model, model_name):
    features = ['female', 'age', 'height', 'mass', 'ta_set', 'rh_set', 'previous_tre_int', 'previous_mtsk_int']
    output = ['tre_int', 'mtsk_int']

    train_df = import_data(features, output)
    features_scaler, output_scaler, train_features, train_output = scale_data(train_df, features, output)
    model = train_model(model, train_features, train_output)

    # Save the model as a pkl file
    joblib.dump(model, 'model_weights/{}.pkl'.format(model_name))

    all_tre_rmse = []
    all_mtsk_rmse = []

    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 1 (prolonged)', 'hot', features, features_scaler, output_scaler, model, model_name)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 2 (indoor)', 'cool', features, features_scaler, output_scaler, model, model_name)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 2 (indoor)', 'temp', features, features_scaler, output_scaler, model, model_name)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 2 (indoor)', 'warm', features, features_scaler, output_scaler, model, model_name)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 2 (indoor)', 'hot', features, features_scaler, output_scaler, model, model_name)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 3 (cooling)', 'hot', features, features_scaler, output_scaler, model, model_name)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)

    avg_tre_rmse = np.mean(all_tre_rmse)
    avg_mtsk_rmse = np.mean(all_mtsk_rmse)

    print(f"{model_name}")
    print("Average TRE RMSE:", avg_tre_rmse)
    print("Average MTSK RMSE:", avg_mtsk_rmse)