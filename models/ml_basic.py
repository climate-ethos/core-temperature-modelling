import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from helpers import get_sample
import joblib


def import_data():
    # Import dataset
    df = pd.read_csv('./dataset/230322_OlderPredictTc_data_thermal.csv')

    # Only use previous values from same individual
    df['previous_tre_int'] = df.groupby('id_all')['tre_int'].shift(1)
    df['previous_mtsk_int'] = df.groupby('id_all')['mtsk_int'].shift(1)

    # Select only time > 0
    df = df[df.time > 0]

    # Select only features and output
    features = ['female', 'age', 'height', 'mass', 'ta_set', 'rh_set', 'previous_tre_int', 'previous_mtsk_int']
    output = ['tre_int', 'mtsk_int']
    df = df[features + output + ['id_all', 'study', 'condition', 'time']]

    # Create train_df based on participants assigned to training set
    train_ids = [46, 34, 68, 30, 40, 98, 89, 65, 24, 58, 85, 67, 28, 39, 35, 77, 26,
                 80, 70, 37, 52, 56, 74, 78, 71, 60, 86, 43, 91, 82, 22, 59, 21, 87,
                 95, 66, 44, 25, 76, 94, 53, 32, 73, 23, 49]
    train_df = df[df['id_all'].isin(train_ids)]

    return train_df, features, output

def train_model(model, train_df, features, output):
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    features_scaler = MinMaxScaler(feature_range=(0,1))
    output_scaler = MinMaxScaler(feature_range=(0,1))

    # Fit scalers
    train_features = features_scaler.fit_transform(train_df[features])
    train_output = output_scaler.fit_transform(train_df[output])

    model.fit(train_features, train_output)

    return model, features_scaler, output_scaler

# SIMULATE
# Simulate initial 60 minutes
def simulate_initial(row, features, features_scaler, output_scaler, model):
    # Initial ambient conditions for cooling environment (22°C 9% RH)
    initial_ambient = [22, 9]
    # Initial core and skin temp
    body_conditions = [37, 32]
    # Body parameters taken from data
    body_parameters = row[features[:-4]].tolist()
    # Simulate for 60 mins
    for i in range(60):
        X = np.array(body_parameters + initial_ambient + body_conditions).reshape(1, -1)
        X = features_scaler.transform(X)
        y = model.predict(X)
        y = output_scaler.inverse_transform(y)
        body_conditions = y[0].tolist()
    # Return core and skin temp at end of 60 mins (e.g. [37, 32])
    return body_conditions

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
                predicted_values = [simulate_initial(row, features, features_scaler, output_scaler, model)]
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
    train_df, features, output = import_data()
    model, features_scaler, output_scaler = train_model(model, train_df, features, output)

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