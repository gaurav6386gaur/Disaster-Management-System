import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load trained model
model = joblib.load("disaster_model_balanced.pkl")

# Connect to SQLite database for prediction history
conn = sqlite3.connect("prediction_history.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                temperature REAL,
                humidity REAL,
                rainfall REAL,
                wind_speed REAL,
                region_type TEXT,
                month INTEGER,
                earthquake_magnitude REAL,
                forest_density REAL,
                past_disaster_count INTEGER,
                population_density REAL,
                prediction TEXT,
                prediction_type TEXT,
                timestamp TEXT
            )''')
conn.commit()

# Sidebar Navigation
st.set_page_config(page_title="Disaster Prediction System", layout="wide")

st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #d7f5d7;  /* Light blue */
        padding: 22px 15px;
    }

    /* Sidebar text elements */
    [data-testid="stSidebar"] .css-10trblm,
    [data-testid="stSidebar"] .css-1v0mbdj,
    [data-testid="stSidebar"] .css-1d391kg {
        color: black !important;
        font-weight: bold !important;
        font-family: Georgia, serif;
        font-size: 18px;
        line-height: 2;
        margin-bottom: 18px;
    }
    </style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.title("🌍 Navigation")
    main_page = st.radio("Select Main Section", ["Home", "Prediction", "Evaluation", "History", "About"])

# Region map
region_map = {"Forest": 0, "Coastal": 1, "Rural": 2, "Urban": 3}
reverse_region_map = {v: k for k, v in region_map.items()}

if main_page == "Home":
    st.markdown("""
    <div style="background-color:#1e2f40;padding:25px 30px;border-radius:10px;margin-bottom:25px;">
        <h1 style="color:white;font-family:'Segoe UI',sans-serif;">
            🏠 Welcome to the Disaster Type Prediction System
        </h1>
        <p style="color:white;font-size:16px;line-height:1.6;font-family:'Segoe UI',sans-serif;">
            This system uses <b>machine learning</b> to predict the type of disaster —
            <span style="color:#ffd700;">Flood</span>, <span style="color:#ff6347;">Earthquake</span>,
            <span style="color:#ffa500;">Fire</span>, or <span style="color:#00fa9a;">No Disaster</span> —
            based on environmental and geographical input data.
        </p>
    </div>
""", unsafe_allow_html=True)

    st.markdown("#### 🎥 Related video for Disasters")
    st.markdown("""
<iframe width="800" height="415" src="https://www.youtube.com/embed/jhRuUoTnA6g?si=MKRP8ifmqPv5P_Xh" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""", unsafe_allow_html=True)

elif main_page == "Prediction":
    sub_page = st.selectbox("Select Prediction Type", ["Single Prediction", "Batch Prediction"])

    if sub_page == "Single Prediction":
        st.title("🔍 Single Prediction")
        with st.form("prediction_form"):
            temperature = st.number_input("🌡️ Temperature (°C)", 0.0, 60.0, step=0.1)
            humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, step=1.0)
            rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, step=1.0)
            wind_speed = st.number_input("💨 Wind Speed (km/h)", 0.0, 150.0, step=1.0)
            region_type = st.selectbox("📍 Region Type", list(region_map.keys()))
            month = st.selectbox("🗓️ Month", list(range(1, 13)))
            earthquake_magnitude = st.number_input("🌐 Earthquake Magnitude", 0.0, 10.0, step=0.1)
            forest_density = st.number_input("🌲 Forest Density (%)", 0.0, 100.0, step=1.0)
            past_disaster_count = st.number_input("📊 Past Disaster Count", 0, 20, step=1)
            population_density = st.number_input("🏘️ Population Density (per km²)", 0.0, 10000.0, step=10.0)

            submitted = st.form_submit_button("Predict")

        if submitted:
            region_encoded = region_map[region_type]
            input_df = pd.DataFrame([[
                temperature, humidity, rainfall, wind_speed, region_encoded, month,
                earthquake_magnitude, forest_density, past_disaster_count, population_density
            ]], columns=[
                "temperature", "humidity", "rainfall", "wind_speed", "region_type", "month",
                "earthquake_magnitude", "forest_density", "past_disaster_count", "population_density"
            ])
            prediction = model.predict(input_df)[0]
            st.success(f"🧭 Predicted Disaster Type: {prediction}")

            # Log prediction to database
            c.execute("""
                INSERT INTO predictions (temperature, humidity, rainfall, wind_speed, region_type, month,
                earthquake_magnitude, forest_density, past_disaster_count, population_density, prediction, prediction_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (temperature, humidity, rainfall, wind_speed, region_type, month,
                  earthquake_magnitude, forest_density, past_disaster_count, population_density, prediction, "Single", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()

    elif sub_page == "Batch Prediction":
        st.title("📁 Batch Prediction")
        uploaded_file = st.file_uploader("Upload a CSV file with 10 columns (no header)", type="csv")
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file, header=None)
                input_df.columns = [
                    "temperature", "humidity", "rainfall", "wind_speed", "region_type", "month",
                    "earthquake_magnitude", "forest_density", "past_disaster_count", "population_density"
                ]
                if input_df["region_type"].dtype == object:
                    input_df["region_type"] = input_df["region_type"].map(region_map)
                predictions = model.predict(input_df)
                input_df["Prediction"] = predictions
                input_df["Prediction_Type"] = "Batch"
                input_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Log each prediction to the database
                for _, row in input_df.iterrows():
                    c.execute("""
                        INSERT INTO predictions (temperature, humidity, rainfall, wind_speed, region_type, month,
                        earthquake_magnitude, forest_density, past_disaster_count, population_density, prediction, prediction_type, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row.temperature, row.humidity, row.rainfall, row.wind_speed, int(row.region_type), row.month,
                          row.earthquake_magnitude, row.forest_density, row.past_disaster_count,
                          row.population_density, row.Prediction, row.Prediction_Type, row.Timestamp))
                conn.commit()

                st.success("✅ Batch prediction complete.")
                st.dataframe(input_df)
            except Exception as e:
                st.error(f"Error processing file: {e}")

elif main_page == "Evaluation":
    st.title("📊 Model Evaluation Dashboard")

    eval_option = st.selectbox("Choose Evaluation Component", [
        "Accuracy Score",
        "Classification Report",
        "Confusion Matrix",
        "Prediction Distribution",
        "Feature Importances"
    ])

    df = pd.read_csv("disaster_dataset_balanced.csv")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    label_enc = LabelEncoder()
    df["region_type"] = label_enc.fit_transform(df["region_type"])
    X = df.drop("Disaster_Type", axis=1)
    y = df["Disaster_Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    if eval_option == "Accuracy Score":
        acc = accuracy_score(y_test, y_pred)
        st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")

    elif eval_option == "Classification Report":
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap="Blues"))

    elif eval_option == "Confusion Matrix":
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=model.classes_, yticklabels=model.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    elif eval_option == "Prediction Distribution":
        pred_series = pd.Series(y_pred)
        pred_counts = pred_series.value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="viridis")
        ax.set_title("Distribution of Predicted Classes")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif eval_option == "Feature Importances":
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="crest")
        ax.set_title("Feature Importances")
        st.pyplot(fig)                

elif main_page == "History":
    st.title("🕘 Prediction History")
    history_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
    if not history_df.empty:
        st.dataframe(history_df)
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download History as CSV", csv, "prediction_history.csv", "text/csv")
    else:
        st.info("No prediction history found.")

elif main_page == "About":
    st.markdown("""
        <style>
            .about-box {
                background-color: #1e2f40;
                padding: 25px 30px;
                border-radius: 12px;
                margin-bottom: 25px;
                color: white;
                font-family: 'Segoe UI', sans-serif;
                line-height: 1.8;
                font-size: 16px;
            }
            .about-box h2 {
                color: #00f9b4;
                font-size: 28px;
                margin-bottom: 10px;
            }
            .about-box h4 {
                color: #f9e154;
                margin-top: 25px;
                margin-bottom: 10px;
            }
            .about-box ul {
                list-style-type: square;
                padding-left: 20px;
            }
            .about-box blockquote {
                font-style: italic;
                color: #9fc9ff;
                border-left: 4px solid #00f9b4;
                padding-left: 10px;
                margin: 20px 0;
            }
        </style>

        <div class="about-box">
            <h2>🧭 About This Project</h2>
            <p>
                The <strong>Disaster Type Prediction System</strong> is an intelligent machine learning-powered application
                built to classify disasters (Flood, Fire, Earthquake, No Disaster) based on real-world environmental input.
            </p>
            <h4>👩‍💻 Developed By</h4>
            <p>Team <strong>Champs</strong>: Gaurav, Divyanshu, Ankit</p>
            <h4>🏫 Guided By</h4>
            <p><strong>DR. Sangita Mishra Ma'am</strong><br>BBDEC (Affiliated to AKTU)</p>
            <h4>🛠️ Technologies Used</h4>
            <ul>
                <li>Python, Streamlit</li>
                <li>Scikit-learn (RandomForestClassifier)</li>
                <li>Pandas, NumPy, Matplotlib, Seaborn</li>
                <li>SQLite for prediction history</li>
            </ul>
            <h4>🌟 Key Features</h4>
            <ul>
                <li>Single & Batch Predictions</li>
                <li>Evaluation Dashboard (Confusion Matrix, Accuracy, Feature Importance)</li>
                <li>User-friendly Web Interface</li>
                <li>Real-time Logging and Downloadable History</li>
            </ul>
            <h4>🚀 Future Scope</h4>
            <ul>
                <li>Real-time sensor data integration</li>
                <li>User login system</li>
                <li>Mobile app deployment</li>
                <li>Automated disaster alerts</li>
            </ul>
            <blockquote>
                This project was built with support from Streamlit community.
            </blockquote>
        </div>
                """, unsafe_allow_html=True )



# Stylish footer using HTML and CSS
# Stylish green footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #2e8b57;  /* Medium green */
            color: white;
            text-align: center;
            padding: 12px;
            font-size: 15px;
            font-family: cursive;
            letter-spacing: 0.5px;
            box-shadow: 0 -1px 5px rgba(0,0,0,0.2);
        }
    </style>
    <div class="footer">
        🚨 Developed by Team Champs | BBDEC | Guided by DR. Sangita Mishra Ma'am
    </div>
""", unsafe_allow_html=True)

