import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium, folium_static


BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.title("Next Location Prediction Using the Microsoft GeoLife Dataset")
st.write("""
This app allows users to conduct a basic level pattern-of-life analysis and leverage semi-supervised machine learning techniques
to conduct next location prediction given a previous location and datetime information.
""")

st.sidebar.title("User Inputs")
st.sidebar.subheader("Select A User and Specify Parameters")
with st.sidebar:
    with st.form('Cluster Inputs'):
        approved_uid = ['000', '002', '003', '004', '011', '014']
        uid = st.selectbox(label='User ID', options=approved_uid)
        distance = st.slider(label='Max Distance Between Two Points (in kms)', min_value=.01, max_value=.5, value=.2)
        min_k = st.slider(label='Min. Number of Observations', min_value=1, max_value=5, value=1)
        cluster_submit = st.form_submit_button("Cluster", type="primary")

if cluster_submit:

    payload = {
        "uid": uid,
        "distance": distance,
        "min_k": min_k
    }

    with st.spinner("Clustering..."):
        try:
            cluster_response = requests.post(f"{BACKEND_URL}/cluster", json=payload)
            cluster_response.raise_for_status()  # Raise exception for HTTP errors
            cluster_api_dict = cluster_response.json()
            cluster_df = pd.DataFrame(cluster_api_dict['df'])
            cluster_scores = pd.DataFrame().from_dict(cluster_api_dict['scores'], orient='index').T        
            st.dataframe(cluster_scores, hide_index=True)

            m = folium.Map(
                location=[cluster_df['lat_origin'].median(), cluster_df['lng_origin'].median()], 
                zoom_start=12,
                tiles='OpenStreetMap'
                )

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['lat_origin'], row['lng_origin']],
                    popup=row['cluster_origin'],
                    tooltip=row['cluster_origin'],
                    color="#061C80FF",
                    fill=True,
                    radius=7,
                    fill_color="#061C80FF",
                    fill_opacity=1
                    ).add_to(m)

            for index, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=[row['cluster_dest_lat'], row['cluster_dest_lng']],
                    popup=row['cluster_dest'],
                    tooltip=row['cluster_dest'],
                    color="#A51D1DFF",
                    fill=True,
                    radius=1,
                    fill_color="#A51D1DFF",
                    fill_opacity=1
                    ).add_to(m)
            m.add_child(folium.ClickForLatLng())
            folium_static(m, width=1600, height=600)
            st.dataframe(cluster_df, hide_index=True, width=1600)

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to prediction service: {str(e)}")
            st.warning(f"Make sure the backend service is running at {BACKEND_URL}")


# # Add information about the app
# st.sidebar.header("about")
# st.sidebar.write("""
# Uses a random forest regressor to predict car mpg

# """)

# st.sidebar.header("Feature Impact on MPG")
# feature_importance = pd.DataFrame({
#     'Feature': ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin'],
#     'Importance': [0.12, 0.18, 0.15, 0.25, 0.05, 0.15, 0.10]  # Example values
# })

# st.sidebar.bar_chart(feature_importance.set_index('Feature'))