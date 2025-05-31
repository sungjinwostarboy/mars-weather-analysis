import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def regression_and_visualization(df):
    """
    Analisis regresi polinomial derajat 2 & visualisasi lengkap.
    """
    print("Memulai analisis regresi dan visualisasi...")

    # Fitur dan target
    X = df[['max_temp_C', 'min_temp_C', 'pressure_Pa', 'wind_speed_mps', 'wind_direction_degrees']]
    y = df['avg_temp_C']

    # Split data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Transformasi fitur ke polinomial derajat 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Latih model regresi linier
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Prediksi data uji
    y_pred = model.predict(X_test_poly)

    # Evaluasi model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Koefisien Determinasi (R^2): {r2:.3f}")

    # Visualisasi Prediksi vs Aktual
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Suhu Aktual (°C)')
    plt.ylabel('Suhu Prediksi (°C)')
    plt.title('Prediksi vs Aktual Suhu Mars (Test Set)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualisasi Tren Suhu Mars dari waktu ke waktu
    plt.figure(figsize=(10,5))
    sns.lineplot(data=df, x='terrestrial_date', y='avg_temp_C', marker='o')
    plt.xlabel('Tanggal (Bumi)')
    plt.ylabel('Suhu Rata-rata (°C)')
    plt.title('Tren Suhu Rata-rata Mars berdasarkan Data InSight')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualisasi Distribusi Suhu Rata-rata Mars
    plt.figure(figsize=(8,5))
    sns.histplot(df['avg_temp_C'], bins=15, kde=True)
    plt.xlabel('Suhu Rata-rata (°C)')
    plt.title('Distribusi Suhu Rata-rata Mars')
    plt.tight_layout()
    plt.show()
