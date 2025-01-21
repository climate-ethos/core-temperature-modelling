from models.ml_basic import import_data_all, scale_data, simulate_initial
import numpy as np
import joblib
import os

model_name = 'ml_ridge_regression'  # Replace with the desired model name

# Save scalers
def save_scalers(features_scaler, output_scaler, directory='model_weights/scalers'):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save the scalers
    joblib.dump(features_scaler, os.path.join(directory, 'features_scaler.pkl'))
    joblib.dump(output_scaler, os.path.join(directory, 'output_scaler.pkl'))

# Load scalers
def load_scalers(directory='model_weights/scalers'):
    # Load the scalers
    features_scaler = joblib.load(os.path.join(directory, 'features_scaler.pkl'))
    output_scaler = joblib.load(os.path.join(directory, 'output_scaler.pkl'))

    return features_scaler, output_scaler

# Define the features and output variables
features = ['female', 'age', 'height', 'mass', 'ta_set', 'rh_set', 'previous_tre_int', 'previous_mtsk_int']
output = ['tre_int', 'mtsk_int']

# Create scalers the same as for training
train_df = import_data_all(features, output)
features_scaler, output_scaler, train_features, train_output = scale_data(train_df, features, output)
# Save them so they can be used later
save_scalers(features_scaler, output_scaler)

# Function to predict tre and mtsk for custom input features
def predict_custom_input(fold, female, age, height, mass, ta_set, rh_set, time_steps=540):
    # Load model
    model = joblib.load(f'model_weights/{model_name}-fold{fold}.pkl')
    # Get parameters
    body_parameters = [female, age, height, mass]
    ambient_conditions = [ta_set, rh_set]
    # Get starting core and skin temp
    initial_body_conditions = simulate_initial(body_parameters, features_scaler, output_scaler, model)
    custom_input = body_parameters + ambient_conditions + initial_body_conditions
    # Simulate for time steps
    predicted_values = []
    for _ in range(time_steps):
        X = np.array(custom_input).reshape(1, -1)
        X = features_scaler.transform(X)
        y = model.predict(X)
        y = output_scaler.inverse_transform(y)
        predicted_values.append(y[0].tolist())
        custom_input[-2:] = y[0].tolist()

    # Get the final predicted tre and mtsk values
    final_tre, final_mtsk = predicted_values[-1]
    initial_tre, initial_mtsk = initial_body_conditions
    # Get the initial values
    return final_tre, final_mtsk, initial_tre, initial_mtsk