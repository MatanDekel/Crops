#data_reader if need select
# data filter based on what i need
# send for training the same data
# prediction gets the training results - send an input gets predictions
from project.scripts.data_filter import season_filter, region_filter, season_specific_weight, season_specific_weightD, \
    crop_type
from project.scripts.data_reader import reader
from project.scripts.predict import app
from project.scripts.training import train
import requests


def make_prediction(month, year, crop, season, region):
    url = 'http://127.0.0.1:5000/predict'
    data = {
        'month': month,
        'year': year,
        'crop': crop,
        'season': season,
        'region': region
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        predictions = result['predictions']
        evaluation_results = result['evaluation_results']
        print("Predictions:")
        print(predictions)
        print("Evaluation Results:")
        print(evaluation_results)
    else:
        print(f"Failed to get a response: {response.status_code}")


if __name__ == "__main__":
    app.run(debug=True)
    df = reader(r'D:\Final project\pythonProject2\project\data\data.xlsx')
    for season in df['season'].unique():
        for region in df['region'].unique():
            for crop in df['desc'].unique():
                filtered_df = df
                filtered_df = crop_type(filtered_df, crop)
                filtered_df = season_filter(filtered_df, season)
                filtered_df = region_filter(filtered_df, region)
                filtered_df = season_specific_weight(filtered_df, season)
                train(filtered_df, region, season, crop)
                filtered_df = season_specific_weightD(filtered_df, season)
