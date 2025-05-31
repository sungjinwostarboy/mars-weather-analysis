import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def get_mars_weather_data(api_key):
    url = f"https://api.nasa.gov/insight_weather/?api_key={api_key}&feedtype=json&ver=1.0"
    response = requests.get(url)
    data = response.json()
    sols = data.get('sol_keys', [])
    records = []
    for sol in sols:
        sol_data = data[sol]
        records.append({
            'sol': sol,
            'terrestrial_date': sol_data.get('First_UTC', None),
            'avg_temp_C': sol_data.get('AT', {}).get('av', np.nan),
            'min_temp_C': sol_data.get('AT', {}).get('mn', np.nan),
            'max_temp_C': sol_data.get('AT', {}).get('mx', np.nan),
            'pressure_Pa': sol_data.get('PRE', {}).get('av', np.nan),
            'wind_speed_mps': sol_data.get('HWS', {}).get('av', np.nan),
            'wind_direction_degrees': sol_data.get('WD', {}).get('most_common', {}).get('compass_degrees', np.nan)
        })
    df = pd.DataFrame(records)
    df['terrestrial_date'] = pd.to_datetime(df['terrestrial_date'])
    return df

def preprocess_mars_data(df):
    df_clean = df.dropna(subset=['avg_temp_C', 'min_temp_C', 'max_temp_C', 'pressure_Pa', 'wind_speed_mps', 'wind_direction_degrees'])
    return df_clean.reset_index(drop=True)

def save_data_to_csv(df, filename="mars_weather_data_clean.csv"):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def regression_and_visualization(df):
    X = df[['max_temp_C', 'min_temp_C', 'pressure_Pa', 'wind_speed_mps', 'wind_direction_degrees']]
    y = df['avg_temp_C']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y, y=y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title("Actual vs Predicted Temperature on Mars")
    plt.show()

def main():
    API_KEY = "rKez6R5M9EIPN0u8euCz3GLakj07xgEuf51CHiA3"
    df_raw = get_mars_weather_data(API_KEY)
    df_clean = preprocess_mars_data(df_raw)
    save_data_to_csv(df_clean)
    regression_and_visualization(df_clean)

if __name__ == "__main__":
    main()
