def preprocess_mars_data(df):
    """
    Membersihkan data: hilangkan baris dengan nilai penting kosong.
    Reset index.
    """
    print("Preprocessing data...")
    initial_len = len(df)
    df_clean = df.dropna(subset=['avg_temp_C', 'min_temp_C', 'max_temp_C', 'pressure_Pa', 'wind_speed_mps', 'wind_direction_degrees'])
    final_len = len(df_clean)
    print(f"Data dibersihkan: {initial_len} → {final_len} record (dibuang {initial_len - final_len})")
    df_clean = df_clean.reset_index(drop=True)
    return df_clean
