import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi ambil data dari NASA Mars InSight API
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

# Fungsi preprocessing data: hapus data yang tidak lengkap
def preprocess_mars_data(df):
    df_clean = df.dropna(subset=['avg_temp_C', 'min_temp_C', 'max_temp_C', 'pressure_Pa', 'wind_speed_mps', 'wind_direction_degrees'])
    return df_clean.reset_index(drop=True)

# Fungsi regresi polinomial dan visualisasi
def regression_and_visualization(df):
    X = df[['max_temp_C', 'min_temp_C', 'pressure_Pa', 'wind_speed_mps', 'wind_direction_degrees']]
    y = df['avg_temp_C']
    
    # Polinomial degree 2 untuk menangkap hubungan kompleks
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    st.write(f"### Model Performance:")
    st.write(f"- Mean Squared Error (MSE): {mse:.3f}")
    st.write(f"- R-squared (R2): {r2:.3f}")
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=y, y=y_pred, ax=ax)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual Avg Temperature (°C)")
    ax.set_ylabel("Predicted Avg Temperature (°C)")
    ax.set_title("Actual vs Predicted Average Temperature on Mars")
    st.pyplot(fig)

# Main UI Streamlit
def main():
    st.title("Mars Weather Data Analysis using NASA InSight API")
    st.write("""
        This app fetches recent weather data from Mars (InSight Lander),
        preprocesses the data, and performs a polynomial regression analysis 
        to predict average temperature based on other weather variables.
    """)
    
    api_key = st.text_input("Enter your NASA API Key:", "rKez6R5M9EIPN0u8euCz3GLakj07xgEuf51CHiA3")
    
    if st.button("Fetch & Analyze Data"):
        with st.spinner("Fetching data from NASA Mars Weather API..."):
            df_raw = get_mars_weather_data(api_key)
        st.success("Data fetched successfully!")
        st.write("### Raw Data:")
        st.dataframe(df_raw)
        
        df_clean = preprocess_mars_data(df_raw)
        st.write("### Cleaned Data (after preprocessing):")
        st.dataframe(df_clean)
        
        regression_and_visualization(df_clean)

if __name__ == "__main__":
    main()
