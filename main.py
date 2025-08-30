import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# ------------------ Page Config ------------------ #
st.set_page_config(
    page_title="Tourist Crowd Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Background & CSS ------------------ #
st.markdown(
    """
    <style>
    body {
        background-image: 
            linear-gradient(rgba(255,255,255,0.2), rgba(255,255,255,0.2)),
            url("https://images.unsplash.com/photo-1516483638261-f4dbaf036963?auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        animation: gradientShift 20s ease infinite;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .appview-container {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 15px;
        padding: 15px;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.75) !important;
        border-radius: 10px;
        padding: 10px;
    }
    table.dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Load Models & Data ------------------ #
@st.cache_resource
def load_models():
    reg_model = joblib.load("regression_model.joblib")
    clf_model = joblib.load("classification_model.joblib")
    le_place = joblib.load("label_encoder_place.joblib")
    le_country = joblib.load("label_encoder_country.joblib")
    le_holiday = joblib.load("label_encoder_holiday.joblib")
    le_crowd = joblib.load("label_encoder_crowd.joblib")
    return reg_model, clf_model, le_place, le_country, le_holiday, le_crowd

@st.cache_data
def load_data():
    df = pd.read_csv("main.2 - Sheet1.csv")
    df.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
    return df

reg_model, clf_model, le_place, le_country, le_holiday, le_crowd = load_models()
df = load_data()

# ------------------ App Title ------------------ #
st.title("🙏 Tourist Crowd Predictor for Religious Places 🙏")
st.subheader("Plan your trip with predicted crowd levels and visitor trends 🌏")

# ------------------ Sidebar Inputs ------------------ #
st.sidebar.header("🔧 Trip & Place Inputs")

religious_place = st.sidebar.selectbox(
    "Select Religious Place", sorted(df["Religious_Place"].unique())
)

available_countries = df[df["Religious_Place"] == religious_place]["Country"].unique()
country = st.sidebar.selectbox("Select Country", sorted(available_countries))

place_data = df[(df["Religious_Place"] == religious_place) & (df["Country"] == country)]

# Pre-fill Public Holiday
if not place_data.empty and "Public_Holiday" in place_data.columns:
    most_common_holiday = place_data["Public_Holiday"].mode()[0]
else:
    most_common_holiday = "No"

public_holiday = st.sidebar.selectbox(
    "Public Holiday?", ["Yes", "No"], index=0 if most_common_holiday == "Yes" else 1
)

# Pre-fill Past Crowd
if not place_data.empty:
    avg_past_crowd = int(place_data["Past_Crowd_Levels"].mean())
else:
    avg_past_crowd = int(df["Past_Crowd_Levels"].median())

past_crowd = st.sidebar.number_input(
    "Past Crowd Level",
    min_value=int(df["Past_Crowd_Levels"].min()),
    max_value=int(df["Past_Crowd_Levels"].max()),
    value=avg_past_crowd
)

# Date Range
date_range = st.sidebar.date_input(
    "Select Trip Date Range",
    value=(datetime.today(), datetime.today() + timedelta(days=3))
)
start_date, end_date = date_range

# ------------------ Predict Button ------------------ #
if st.button("Predict Crowd Levels for Trip"):
    try:
        place_encoded = le_place.transform([religious_place])[0]
        country_encoded = le_country.transform([country])[0]
        holiday_encoded = le_holiday.transform([public_holiday])[0]

        dates = pd.date_range(start=start_date, end=end_date)
        predictions = []

        # Generate predictions
        for single_date in dates:
            day = single_date.day
            month = single_date.month
            weekday = single_date.weekday()
            is_weekend = 1 if weekday >= 5 else 0

            X_new = np.array([[past_crowd, holiday_encoded, day, month, weekday, is_weekend,
                               place_encoded, country_encoded]])

            visitor_count_pred = reg_model.predict(X_new)[0]
            crowd_level_pred = clf_model.predict(X_new)[0]
            crowd_level_label = le_crowd.inverse_transform([crowd_level_pred])[0]

            emoji = "✅" if crowd_level_label.lower()=="low" else "⚠️" if crowd_level_label.lower()=="medium" else "🔥"
            predictions.append({
                "Date": single_date,
                "Visitor_Count": int(visitor_count_pred),
                "Crowd_Level": crowd_level_label,
                "Indicator": emoji
            })

        df_pred = pd.DataFrame(predictions)

        st.success(f"Predictions from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Show table with emoji
        st.dataframe(df_pred.style.apply(
            lambda x: ['background-color: lightgreen' if v=="Low" else 'background-color: yellow' if v=="Medium" else 'background-color: #ff9999' for v in x], subset=["Crowd_Level"]
        ))

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df_pred.set_index("Date")["Visitor_Count"])
        with col2:
            st.bar_chart(df_pred.set_index("Date")["Visitor_Count"])

        # Crowd Level Tips
        st.header("📌 Tips for Your Trip")
        for idx, row in df_pred.iterrows():
            if row["Crowd_Level"].lower() == "high":
                st.error(f"{row['Date'].strftime('%Y-%m-%d')}: 🔥 High crowd. Consider early morning visits or avoid weekends.")
            elif row["Crowd_Level"].lower() == "medium":
                st.warning(f"{row['Date'].strftime('%Y-%m-%d')}: ⚠️ Medium crowd. Plan accordingly.")
            else:
                st.success(f"{row['Date'].strftime('%Y-%m-%d')}: ✅ Low crowd. Perfect day to visit!")

        # Download Option
        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name='trip_crowd_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error: {e}")
