import numpy as np
from tensorflow import keras
from models.ml_basic import import_data_all, scale_data
from models.ml_sequential import preprocess_data, concat_extra_data


# Define the features and output variables
features = ['female', 'age', 'height', 'mass', 'ta_set', 'rh_set']
output = ['tre_int', 'mtsk_int']

# Create scalers the same as for training
train_df = import_data_all(features, output)
train_df = concat_extra_data(train_df)
features_scaler, output_scaler, train_features, train_output = scale_data(train_df, features, output)
_, _, max_len = preprocess_data(train_df, train_features, train_output)

# Function to predict tre and mtsk for custom input features
def predict_custom_input_sequential(model_name, fold, female, age, height, mass, ta_set, rh_set, time_steps=540):
    # Load model
    model = keras.models.load_model(f'model_weights/{model_name}-fold{fold}.keras')
    # Get input
    custom_input = [female, age, height, mass, ta_set, rh_set]
    # Scale the custom input features
    X_scaled = features_scaler.transform([custom_input])

    # Create a sequence of the custom input features
    X_seq = np.repeat(X_scaled, time_steps, axis=0)

    # Pad the sequence to have the same length as the training data
    X_padded = np.pad(X_seq, ((0, max_len - time_steps), (0, 0)), mode='constant')

    # Reshape the padded sequence to match the model's input shape
    X_padded = np.reshape(X_padded, (1, max_len, len(features)))

    # Make predictions
    predictions = model.predict(X_padded, verbose=0)

    # Remove predictions corresponding to padded inputs
    unpadded_predictions = predictions[0, :time_steps]

    # Inverse transform the predictions
    unpadded_predictions = output_scaler.inverse_transform(unpadded_predictions)

    # Get the final predicted tre and mtsk values
    final_tre, final_mtsk = unpadded_predictions[-1]
    return final_tre, final_mtsk