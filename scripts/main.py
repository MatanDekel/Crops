# from app import app
from scripts.data_filter import season_filter, region_filter, season_specific_weight, season_specific_weightD, \
    crop_type
from scripts.data_reader import reader
from scripts.training import train
import requests

from utils.data_utils import import_data


def make_prediction(date, crop, region):
    url = 'https://crops-wrxx.onrender.com/predict'
    data = {
        'date': date,
        'crop': crop,
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
    df = reader(r'data/data.xlsx')
    for season in df['season'].unique():
        for region in df['region'].unique():
            for crop in df['desc'].unique():
                filtered_df = df
                filtered_df = crop_type(filtered_df, crop)
                filtered_df = season_filter(filtered_df, season)
                filtered_df = region_filter(filtered_df, region)
                filtered_df = season_specific_weight(filtered_df, season)
                if len(filtered_df) > 5:
                    import_data(
                        fr"data/filtered_data/filtered_data_{region}_{season}_{crop}.xlsx",
                        filtered_df)
                    train(filtered_df, region, season, crop)
                    filtered_df = season_specific_weightD(filtered_df, season)

    #  app.run(debug=True)
