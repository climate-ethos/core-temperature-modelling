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
    folds = [
        [24, 28, 43, 50, 57, 62, 66, 68, 71, 72, 75, 79, 83, 86, 90, 93, 94, 95, 97],
        [22, 25, 26, 32, 34, 35, 45, 48, 54, 55, 59, 67, 69, 70, 74, 76, 85, 91, 98],
        [21, 27, 30, 37, 42, 46, 47, 49, 53, 61, 63, 65, 73, 78, 81, 82, 87, 92, 96],
        [23, 29, 33, 36, 38, 39, 40, 41, 44, 52, 56, 58, 60, 64, 77, 80, 84, 88, 89]
    ]

    fold_data = []

    for fold in folds:
        # Training data is everything not in the fold
        train_df = df[not df['id_all'].isin(fold)]
        # Reset index
        train_df.reset_index(inplace=True)
        fold_data.append(train_df)

    return fold_data

# Scale the dataset between 0 to 1
def scale_data(train_df, features, output):
    features_scaler = MinMaxScaler(feature_range=(0,1))
    output_scaler = MinMaxScaler(feature_range=(0,1))

    # Use all data to fit scalars so that there is universal scalars
    all_data_df = pd.concat(import_data(features, output))

    # Fit scalers
    features_scaler.fit(all_data_df[features])
    output_scaler.fit(all_data_df[output])

    # Transform training data
    train_features = features_scaler.transform(train_df[features])
    train_output = output_scaler.transform(train_df[output])

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
# Simulate initial 120 minutes
def simulate_initial(body_parameters, features_scaler, output_scaler, model):
    # Initial ambient conditions for cooling environment (23°C 50% RH)
    initial_ambient = [cool_Ta, cool_RH]
    # Initial core and skin temp
    body_conditions = [37, 32]
    # Simulate for 120 mins
    for i in range(120):
        X = np.array(body_parameters + initial_ambient + body_conditions).reshape(1, -1)
        X = features_scaler.transform(X)
        y = model.predict(X)
        y = output_scaler.inverse_transform(y)
        body_conditions = y[0].tolist()
    # Return core and skin temp at end of 120 mins (e.g. [37, 32])
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

    fold_data = import_data(features, output)

    # Loop for each fold_data
    for idx, train_df in enumerate(fold_data):
        print("Fold:", idx+1)
        features_scaler, output_scaler, train_features, train_output = scale_data(train_df, features, output)
        model = train_model(model, train_features, train_output)

        # Save the model as a pkl file
        joblib.dump(model, 'model_weights/{}-fold{}.pkl'.format(model_name, ))

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

def run_all(model_name):
    features = ['female', 'age', 'height', 'mass', 'ta_set', 'rh_set', 'previous_tre_int', 'previous_mtsk_int']
    output = ['tre_int', 'mtsk_int']

    # Import and scale data
    fold_data = import_data(features, output)

    # Loop for each fold_data
    for idx, train_df in enumerate(fold_data):
        print("Fold:", idx+1)
        features_scaler, output_scaler, train_features, train_output = scale_data(train_df, features, output)

        # Load the model from pkl file instead of training
        model = joblib.load('model_weights/{}.pkl'.format(model_name))

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