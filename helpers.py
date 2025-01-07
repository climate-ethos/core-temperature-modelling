import pandas as pd

folds = [
    [24, 28, 43, 50, 57, 62, 66, 68, 71, 72, 75, 79, 83, 86, 90, 93, 94, 95, 97],
    [22, 25, 26, 32, 34, 35, 45, 48, 54, 55, 59, 67, 69, 70, 74, 76, 85, 91, 98],
    [21, 27, 30, 37, 42, 46, 47, 49, 53, 61, 63, 65, 73, 78, 81, 82, 87, 92, 96],
    [23, 29, 33, 36, 38, 39, 40, 41, 44, 52, 56, 58, 60, 64, 77, 80, 84, 88, 89]
]

def get_sample(study, condition):
    df_dataset = pd.read_csv('./dataset/230322_OlderPredictTc_data_thermal.csv')
    sample = df_dataset[df_dataset.study == study]
    sample = sample[sample.condition == condition]
    sample = sample[sample.time > 0]
    # Remove ID 26 (incomplete data) in heatwave 2 hot
    if study == 'heatwave 2 (indoor)' and condition == 'hot':
        sample = sample[sample['id'] != 26]
    return sample