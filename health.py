import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load and preprocess dataset (as done earlier)
df = pd.read_csv("CVD_cleaned.csv")
df = df.dropna(subset=["Heart_Disease"])
df["Heart_Disease"] = df["Heart_Disease"].map({"Yes": 1, "No": 0})
X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]

# Encode categoricals
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X[X.columns] = imputer.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "heart_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(imputer, "imputer.pkl")

# --------- USER INPUT PREDICTION ---------
def predict_user_input():
    model = joblib.load("heart_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    imputer = joblib.load("imputer.pkl")

    input_data = {}

    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            print(f"\n{col} options: {list(options)}")
            value = input(f"Enter {col}: ")
            while value not in options:
                print("Invalid option. Try again.")
                value = input(f"Enter {col}: ")
            value = label_encoders[col].transform([value])[0]
        else:
            value = float(input(f"Enter {col} (numeric): "))
        input_data[col] = value

    # Create DataFrame
    user_df = pd.DataFrame([input_data])
    user_df = pd.DataFrame(imputer.transform(user_df), columns=user_df.columns)

    # Predict
    prediction = model.predict(user_df)[0]
    print("\n🩺 Prediction:", "At Risk of Heart Disease" if prediction == 1 else "No Risk Detected")

# Run prediction
predict_user_input()import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load and preprocess dataset (as done earlier)
df = pd.read_csv("CVD_cleaned.csv")
df = df.dropna(subset=["Heart_Disease"])
df["Heart_Disease"] = df["Heart_Disease"].map({"Yes": 1, "No": 0})
X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]

# Encode categoricals
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X[X.columns] = imputer.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "heart_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(imputer, "imputer.pkl")

# --------- USER INPUT PREDICTION ---------
def predict_user_input():
    model = joblib.load("heart_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    imputer = joblib.load("imputer.pkl")

    input_data = {}

    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            print(f"\n{col} options: {list(options)}")
            value = input(f"Enter {col}: ")
            while value not in options:
                print("Invalid option. Try again.")
                value = input(f"Enter {col}: ")
            value = label_encoders[col].transform([value])[0]
        else:
            value = float(input(f"Enter {col} (numeric): "))
        input_data[col] = value

    # Create DataFrame
    user_df = pd.DataFrame([input_data])
    user_df = pd.DataFrame(imputer.transform(user_df), columns=user_df.columns)

    # Predict
    prediction = model.predict(user_df)[0]
    print("\n🩺 Prediction:", "At Risk of Heart Disease" if prediction == 1 else "No Risk Detected")

# Run prediction
predict_user_input()import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load and preprocess dataset (as done earlier)
df = pd.read_csv("CVD_cleaned.csv")
df = df.dropna(subset=["Heart_Disease"])
df["Heart_Disease"] = df["Heart_Disease"].map({"Yes": 1, "No": 0})
X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]

# Encode categoricals
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X[X.columns] = imputer.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "heart_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(imputer, "imputer.pkl")

# --------- USER INPUT PREDICTION ---------
def predict_user_input():
    model = joblib.load("heart_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    imputer = joblib.load("imputer.pkl")

    input_data = {}

    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            print(f"\n{col} options: {list(options)}")
            value = input(f"Enter {col}: ")
            while value not in options:
                print("Invalid option. Try again.")
                value = input(f"Enter {col}: ")
            value = label_encoders[col].transform([value])[0]
        else:
            value = float(input(f"Enter {col} (numeric): "))
        input_data[col] = value

    # Create DataFrame
    user_df = pd.DataFrame([input_data])
    user_df = pd.DataFrame(imputer.transform(user_df), columns=user_df.columns)

    # Predict
    prediction = model.predict(user_df)[0]
    print("\n🩺 Prediction:", "At Risk of Heart Disease" if prediction == 1 else "No Risk Detected")

# Run prediction
predict_user_input()import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load and preprocess dataset (as done earlier)
df = pd.read_csv("CVD_cleaned.csv")
df = df.dropna(subset=["Heart_Disease"])
df["Heart_Disease"] = df["Heart_Disease"].map({"Yes": 1, "No": 0})
X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]

# Encode categoricals
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X[X.columns] = imputer.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "heart_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(imputer, "imputer.pkl")

# --------- USER INPUT PREDICTION ---------
def predict_user_input():
    model = joblib.load("heart_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    imputer = joblib.load("imputer.pkl")

    input_data = {}

    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            print(f"\n{col} options: {list(options)}")
            value = input(f"Enter {col}: ")
            while value not in options:
                print("Invalid option. Try again.")
                value = input(f"Enter {col}: ")
            value = label_encoders[col].transform([value])[0]
        else:
            value = float(input(f"Enter {col} (numeric): "))
        input_data[col] = value

    # Create DataFrame
    user_df = pd.DataFrame([input_data])
    user_df = pd.DataFrame(imputer.transform(user_df), columns=user_df.columns)

    # Predict
    prediction = model.predict(user_df)[0]
    print("\n🩺 Prediction:", "At Risk of Heart Disease" if prediction == 1 else "No Risk Detected")

# Run prediction
predict_user_input()v
