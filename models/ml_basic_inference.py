from models.ml_basic import import_data, scale_data, simulate_initial
import numpy as np
import joblib

# Load the trained model
model_name = 'ml_ridge_regression'  # Replace with the desired model name
model = joblib.load(f'model_weights/{model_name}.pkl')

# Define the features and output variables
features = ['female', 'age', 'height', 'mass', 'ta_set', 'rh_set', 'previous_tre_int', 'previous_mtsk_int']
output = ['tre_int', 'mtsk_int']

# Create scalars the same as for training
train_df = import_data(features, output)
features_scaler, output_scaler, train_features, train_output = scale_data(train_df, features, output)

# Function to predict tre and mtsk for custom input features
def predict_custom_input(female, age, height, mass, ta_set, rh_set, time_steps=540):
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
    return final_tre, final_mtsk