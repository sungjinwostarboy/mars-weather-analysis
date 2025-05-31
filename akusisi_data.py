import requests
import pandas as pd
import numpy as np

def get_mars_weather_data(api_key):
    """
    Mengambil data cuaca Mars dari NASA InSight API.
    Output: DataFrame.
    """
    print("Mengambil data Mars InSight dari NASA API...")
    url = f"https://api.nasa.gov/insight_weather/?api_key={api_key}&feedtype=json&ver=1.0"
    response = requests.get(url)
    data = response.json()

    sols = data.get('sol_keys', [])
    if not sols:
        print("Data Sol tidak ditemukan.")
        return pd.DataFrame()
    
    records = []
    for sol in sols:
        sol_data = data[sol]
        record = {
            'sol': sol,
            'terrestrial_date': sol_data.get('First_UTC', None),
            'avg_temp_C': sol_data.get('AT', {}).get('av', np.nan),
            'min_temp_C': sol_data.get('AT', {}).get('mn', np.nan),
            'max_temp_C': sol_data.get('AT', {}).get('mx', np.nan),
            'pressure_Pa': sol_data.get('PRE', {}).get('av', np.nan),
            'wind_speed_mps': sol_data.get('HWS', {}).get('av', np.nan),
            'wind_direction_degrees': sol_data.get('WD', {}).get('most_common', {}).get('compass_degrees', np.nan)
        }
        records.append(record)

    df = pd.DataFrame(records)
    df['terrestrial_date'] = pd.to_datetime(df['terrestrial_date'])
    print(f"Berhasil mengambil {len(df)} record data.")
    return df
