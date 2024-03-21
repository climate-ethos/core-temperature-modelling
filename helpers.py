import pandas as pd

def get_sample(study, condition):
    df_dataset = pd.read_csv('./dataset/230322_OlderPredictTc_data_thermal.csv')    
    sample = df_dataset[df_dataset.study == study]
    sample = sample[sample.condition == condition]
    sample = sample[sample.time > 0]
    # Remove ID 26 (incomplete data) in heatwave 2 hot
    if study == 'heatwave 2 (indoor)' and condition == 'hot':
        sample = sample[sample['id'] != 26]
    return sample