import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle, os

# Load data
try:
    df = pd.read_csv(r"D:\Bike sharing rental pridication project\data\BikesharingData.csv", encoding="unicode_escape")
except:
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        "Hour": np.random.randint(0,24,n),
        "Temperature(°C)": np.random.uniform(-10,35,n),
        "Humidity(%)": np.random.randint(10,100,n),
        "Wind speed (m/s)": np.random.uniform(0,8,n),
        "Visibility (10m)": np.random.randint(27,2000,n),
        "Dew point temperature(°C)": np.random.uniform(-30,25,n),
        "Solar Radiation (MJ/m2)": np.random.uniform(0,3.5,n),
        "Rainfall(mm)": np.random.exponential(0.1,n),
        "Snowfall (cm)": np.random.exponential(0.05,n),
        "Seasons": np.random.choice(["Spring","Summer","Autumn","Winter"],n),
        "Holiday": np.random.choice(["Holiday","No Holiday"],n),
        "Functioning Day": "Yes",
        "Month": np.random.randint(1,13,n),
        "Weekday": np.random.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],n),
        "Rented Bike Count": np.random.randint(0,3000,n)
    })

# Encode categorical
for col in ["Seasons","Holiday","Functioning Day","Weekday"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features
features = df.drop(columns=["Rented Bike Count"])
target = df["Rented Bike Count"]

# Train
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 :", r2_score(y_test, pred))

# Save
os.makedirs("model", exist_ok=True)
with open("model/bike_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved ✔")
