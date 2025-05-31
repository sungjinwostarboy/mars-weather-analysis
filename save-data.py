def save_data_to_csv(df, filename="mars_weather_data_clean.csv"):
    """
    Simpan DataFrame ke file CSV.
    """
    df.to_csv(filename, index=False)
    print(f"Data bersih disimpan ke file {filename}")
