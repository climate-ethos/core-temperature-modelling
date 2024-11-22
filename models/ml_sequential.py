from models.ml_basic import cool_Ta, cool_RH, import_data, scale_data
from helpers import get_sample
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Create extra data for each participant
# This simulates sitting in air conditioning for 120 minutes before the trial
def concat_extra_data(input_df):
    extra_data = pd.DataFrame()
    for uid in input_df['unique_id'].unique():
        participant_data = input_df[input_df['unique_id'] == uid].iloc[0]
        new_data = pd.DataFrame({
            'female': [participant_data['female']] * 120,
            'age': [participant_data['age']] * 120,
            'height': [participant_data['height']] * 120,
            'mass': [participant_data['mass']] * 120,
            'ta_set': [cool_Ta] * 120,
            'rh_set': [cool_RH] * 120,
            'tre_int': [participant_data['tre_int']] * 120,
            'mtsk_int': [participant_data['mtsk_int']] * 120,
            'id_all': [participant_data['id_all']] * 120,
            'unique_id': [participant_data['unique_id']] * 120,
            'study': [participant_data['study']] * 120,
            'condition': [participant_data['condition']] * 120,
            'time': list(range(-120, 0))
        })
        extra_data = pd.concat([extra_data, new_data], ignore_index=True)
    return pd.concat([extra_data, input_df], ignore_index=True)

# Break data apart into individual sequences and pad
def preprocess_data(train_df, X_scaled, y_scaled):
    # Create sequences based on unique_id
    unique_ids = train_df['unique_id'].unique()
    X_seq, y_seq = [], []
    for uid in unique_ids:
        seq_data = train_df[train_df['unique_id'] == uid]
        X_seq.append(X_scaled[seq_data.index])
        y_seq.append(y_scaled[seq_data.index])

    # Pad sequences to have the same length
    max_len = max(len(seq) for seq in X_seq)
    print("Max sequence length:", max_len)
    X_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in X_seq])
    y_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in y_seq])

    return X_padded, y_padded, max_len

# Scale and preprocess data
def prepare_data(features):
    output = ['tre_int', 'mtsk_int']

    train_df = import_data(features, output)
    train_df = concat_extra_data(train_df)
    features_scaler, output_scaler, train_features, train_output = scale_data(train_df, features, output)
    X_padded, y_padded, max_len = preprocess_data(train_df, train_features, train_output)
    return features_scaler, output_scaler, X_padded, y_padded, max_len

# Run and save an individual trial
def run_and_save_trial(study, condition, features, features_scaler, output_scaler, model, model_name, max_len, is_transformer):
    # Get sample
    sample = get_sample(study, condition)

    # Unique ID to identify an individual
    sample['unique_id'] = sample['study'].astype(str) + '_' + sample['condition'].astype(str) + '_' + sample['id_all'].astype(str)

    sample_extra_data = concat_extra_data(sample)

    # Fit scalers
    all_X_scaled = features_scaler.fit_transform(sample_extra_data[features])

    # Create sequences based on unique_id
    all_unique_ids = sample_extra_data['unique_id'].unique()
    all_X_seq = []
    seq_lengths = []  # Store the original sequence lengths

    for uid in all_unique_ids:
        seq_data = sample_extra_data['unique_id'] == uid
        data_for_uid = all_X_scaled[seq_data]
        all_X_seq.append(data_for_uid)
        seq_lengths.append(len(data_for_uid))  # Store the original sequence length

    # Pad sequences to have the same length
    all_X_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant') for seq in all_X_seq])

    # Make predictions
    if is_transformer:
        predictions = model.predict([all_X_padded, all_X_padded], verbose=0)
    else:
        predictions = model.predict(all_X_padded, verbose=0)

    # Remove predictions corresponding to padded inputs and extra data
    unpadded_predictions = []
    for i, length in enumerate(seq_lengths):
        unpadded_predictions.append(predictions[i, 120:length])  # Slice to remove 120 mins of extra data

    # Flatten the unpadded predictions
    unpadded_predictions = np.concatenate(unpadded_predictions, axis=0)

    # Inverse transform the predictions
    unpadded_predictions = output_scaler.inverse_transform(unpadded_predictions)

    all_core_temps = unpadded_predictions[:, 0]
    all_skin_temps = unpadded_predictions[:, 1]

    # Save to csv
    df = pd.DataFrame(all_core_temps, columns=["tre_predicted"])
    df["mtsk_predicted"] = all_skin_temps
    df.to_csv('results/{}-{}-{}.csv'.format(model_name, study, condition), index=False)
    # Calculate RMSE
    tre_rmse = np.sqrt(mean_squared_error(sample['tre_int'], df['tre_predicted']))
    mtsk_rmse = np.sqrt(mean_squared_error(sample['mtsk_int'], df['mtsk_predicted']))
    return tre_rmse, mtsk_rmse

# Run all trials
def run_all(model, model_name, features, features_scaler, output_scaler, max_len, is_transformer=False):
    all_tre_rmse = []
    all_mtsk_rmse = []

    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 1 (prolonged)', 'hot', features, features_scaler, output_scaler, model, model_name, max_len, is_transformer)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 2 (indoor)', 'cool', features, features_scaler, output_scaler, model, model_name, max_len, is_transformer)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 2 (indoor)', 'temp', features, features_scaler, output_scaler, model, model_name, max_len, is_transformer)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 2 (indoor)', 'warm', features, features_scaler, output_scaler, model, model_name, max_len, is_transformer)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 2 (indoor)', 'hot', features, features_scaler, output_scaler, model, model_name, max_len, is_transformer)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)
    tre_rmse, mtsk_rmse = run_and_save_trial('heatwave 3 (cooling)', 'hot', features, features_scaler, output_scaler, model, model_name, max_len, is_transformer)
    all_tre_rmse.append(tre_rmse)
    all_mtsk_rmse.append(mtsk_rmse)

    avg_tre_rmse = np.mean(all_tre_rmse)
    avg_mtsk_rmse = np.mean(all_mtsk_rmse)

    print(f"{model_name}")
    print("Average TRE RMSE:", avg_tre_rmse)
    print("Average MTSK RMSE:", avg_mtsk_rmse)