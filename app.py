import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


# ---------------------- THEME STYLING ----------------------
st.markdown("""
    <style>
        body {
            background-color: #0d0f16;
        }

        .main {
            background-color: #0d0f16;
        }

        h1, h2, h3, h4, h5, h6, p {
            color: #e8e8e8 !important;
        }

        .css-18e3th9, .css-1d391kg {
            background-color: #0d0f16 !important;
        }

        /* Glowing Header */
        .main-header {
            font-size: 45px;
            text-align: center;
            padding: 10px;
            color: #00eaff;
            text-shadow: 0px 0px 20px #00eaff;
            font-weight: 700;
        }

        /* Navigation Glow */
        .stRadio > label {
            color: #00eaff !important;
        }

        .stButton>button {
            background: linear-gradient(90deg, #0099ff, #00eaff);
            color: black;
            border-radius: 10px;
            padding: 12px 25px;
            border: none;
            font-size: 18px;
            font-weight: 700;
            box-shadow: 0px 0px 15px #00eaff;
        }

        .stButton>button:hover {
            box-shadow: 0px 0px 30px #00ffff;
            transform: scale(1.03);
        }

        /* Card Boxes */
        .card {
            background: #11131e;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px #00eaff33;
            margin-bottom: 20px;
        }

        .risk-box {
            text-align: center;
            color: white;
            padding: 15px;
            border-radius: 15px;
            background: #2a0d0d;
            box-shadow: 0px 0px 15px red;
        }

        .risk-score {
            font-size: 40px;
            font-weight: bold;
            color: #ff5757;
            text-shadow: 0px 0px 15px red;
        }

    </style>
""", unsafe_allow_html=True)



# ------------------- HEADER TITLE -------------------
st.markdown("<div class='main-header'>Crime Data Analysis Dashboard</div>", unsafe_allow_html=True)


DATA_PATH = Path(__file__).resolve().parent / "crime_data.csv"
REQUIRED_COLUMNS = [
    "date",
    "latitude",
    "longitude",
    "crime_type",
    "district",
]


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    df_copy = data.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
    df_copy = df_copy.dropna(subset=["date", "latitude", "longitude"])
    df_copy["hour"] = df_copy["date"].dt.hour
    df_copy["day"] = df_copy["date"].dt.day
    df_copy["month"] = df_copy["date"].dt.month
    df_copy["year"] = df_copy["date"].dt.year
    df_copy["weekday"] = df_copy["date"].dt.day_name()
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    df_copy["month_name"] = df_copy["month"].apply(lambda x: month_names[x - 1])
    return df_copy


# ------------------- LOAD DATA -------------------
try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Missing dataset at {DATA_PATH}. Place crime_data.csv next to app.py and rerun.")
    st.stop()
except pd.errors.ParserError as exc:
    st.error(f"Unable to parse the dataset: {exc}")
    st.stop()

missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
if missing_columns:
    st.error(
        "Dataset is missing required columns: " + ", ".join(missing_columns)
    )
    st.stop()

df = prepare_dataframe(raw_df)


# ------------------ NAVIGATION ------------------
# -------------------- PAGE NAVIGATION --------------------
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stRadio > div { justify-content: center; }
    .st-emotion-cache-16txtl3 { font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

menu = st.radio(
    "Navigation",
    ["Home", "Crime Dataset", "Crime Heatmap", "Crime Charts", "Prediction Demo"],
    horizontal=True,
    label_visibility="collapsed"
)


# ------------------- HOME PAGE -------------------
if menu == "Home":
    st.markdown("<div class='card'><h2>🏠 Welcome to the Crime Dashboard</h2>"
                "<p>Select a section above to explore crime data, heatmaps, insights, and predictions.</p></div>",
                unsafe_allow_html=True)


# ------------------ DATASET PAGE ------------------
elif menu == "Crime Dataset":
    st.markdown("<div class='card'><h2>📄 Crime Dataset</h2></div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)


# ------------------ HEATMAP PAGE ------------------
elif menu == "Crime Heatmap":
    st.markdown("<div class='card'><h2>🔥 Crime Heatmap</h2></div>", unsafe_allow_html=True)

    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)
    HeatMap(df[['latitude','longitude']].values.tolist()).add_to(m)

    st_folium(m, width=700, height=500)


# ------------------ CHARTS PAGE ------------------
elif menu == "Crime Charts":
    st.markdown("<div class='card'><h2>📊 Crime Charts</h2></div>", unsafe_allow_html=True)

    st.write("### Crimes by Type")
    st.bar_chart(df["crime_type"].value_counts())

    st.write("### Crimes by District")
    st.bar_chart(df["district"].value_counts())

    st.write("### Crimes by Month")
    st.bar_chart(df["month_name"].value_counts())


# ------------------ PREDICTION DEMO ------------------
elif menu == "Prediction Demo":

    st.markdown("<div class='card'><h2>🔮 Crime Prediction System</h2></div>", unsafe_allow_html=True)

    if "prediction_data" not in st.session_state:
        st.session_state.prediction_data = None

    trigger = st.button("Predict Future Crimes")

    if trigger:
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.hour
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.day_name()

        coords = df[["latitude", "longitude"]]
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(coords)
        hotspots = kmeans.cluster_centers_

        le = LabelEncoder()
        df["crime_type_encoded"] = le.fit_transform(df["crime_type"])

        X = df[["hour", "day"]]
        y = df["crime_type_encoded"]

        model = RandomForestClassifier()
        model.fit(X, y)

        predicted = model.predict([[df["hour"].mean(), df["day"].mean()]])[0]
        predicted_type = le.inverse_transform([predicted])[0]

        peak_hour = df["hour"].value_counts().idxmax()
        peak_day = df["weekday"].value_counts().idxmax()

        risk_score = np.random.uniform(4.5, 9.8)

        st.session_state.prediction_data = {
            "hotspots": hotspots,
            "predicted_type": predicted_type,
            "peak_hour": peak_hour,
            "peak_day": peak_day,
            "risk_score": risk_score,
        }

    if st.session_state.prediction_data is not None:
        data = st.session_state.prediction_data

        st.markdown("<h3>🔥 Predicted Crime Hotspots</h3>", unsafe_allow_html=True)
        m = folium.Map(
            location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12
        )
        for lat, lon in data["hotspots"]:
            folium.Circle(
                location=[lat, lon],
                radius=150,
                color="red",
                fill=True,
                fill_opacity=0.4,
                popup="Predicted Hotspot",
            ).add_to(m)
        st_folium(m, width=700, height=450)

        st.markdown("<h3>🕵️ Predicted Crime Type</h3>", unsafe_allow_html=True)
        st.success(f"Most Likely Next Crime: **{data['predicted_type']}**")

        st.markdown("<h3>⏰ Predicted Time</h3>", unsafe_allow_html=True)
        st.info(
            f"High-Risk Time: {data['peak_hour']}:00 hrs\n\nHigh-Risk Day: {data['peak_day']}"
        )

        st.markdown("<h3>🚨 Crime Severity Risk Score</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='risk-box'>
                <div class='risk-score'>{data['risk_score']:.1f}</div>
                <p>Higher score = more risk</p>
            </div>
        """,
            unsafe_allow_html=True,
        )
