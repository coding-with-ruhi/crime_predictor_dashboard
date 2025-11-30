#  Crime Data Analysis Dashboard

A Streamlit-based dashboard to visualize, analyze, and predict crime patterns using maps, charts, clustering, and machine learning.

##  Features
- View complete crime dataset
- Crime heatmap using Folium
- Charts for crime type, district, and month
- Prediction demo:
  - Hotspot detection (KMeans)
  - Crime type prediction (Random Forest)
  - High-risk hour/day
  - Risk severity score

## Project Structure
```
project/
│── app.py
│── crime_data.csv
│── README.md
```

##  Installation
Install all dependencies:

```bash
pip install streamlit pandas folium streamlit-folium numpy scikit-learn
```

##  Run the App
```bash
streamlit run app.py
```

Ensure **crime_data.csv** is in the same folder as `app.py`.

##  Dataset Requirements
Your CSV file must contain the following columns:

- `date`
- `latitude`
- `longitude`
- `crime_type`
- `district`

## Prediction Logic (Simplified)
- **Hotspots:** KMeans clusters high-density crime locations  
- **Crime Type Prediction:** Random Forest predicts crime type based on hour/day  
- **Risk Score:** A simulated severity score for display  

