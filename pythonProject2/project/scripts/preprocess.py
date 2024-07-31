from project.utils.data_utils import load_data, preprocess_data

if __name__ == "__main__":
    file_path = r"D:\Final project\pythonProject2\data\newDataV3.xlsx"
    df = load_data(file_path)
    df = preprocess_data(df)

    preprocessed_file_path = r"D:\Final project\pythonProject2\data\preprocessed_data.csv"
    df.to_csv(preprocessed_file_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_file_path}")
