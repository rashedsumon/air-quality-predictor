import streamlit as tf
import os
import pandas as pd
import joblib

# Use local structural component imports
from data_loader import load_pollution_data
from model import train_aqi_model

st.set_page_config(page_title="Global Urban Air Quality Predictor", page_icon="🌍", layout="wide")

st.title("🌍 Global Urban Air Quality & Pollution Analytics")
st.write("An implementation using automated data download streaming from Kagglehub.")

# 1. Access Data Engine
@st.cache_data(show_spinner="Downloading Global Pollution Data from Kaggle... (Please wait)")
def get_cached_data():
    return load_pollution_data()

try:
    df = get_cached_data()
    st.success("Dataset pulled and synced smoothly from Kagglehub storage layers!")
    
    # Layout splits
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Dataset Metrics")
        st.metric("Total Records Tracked", f"{df.shape[0]:,}")
        st.metric("Feature Columns Found", f"{df.shape[1]}")
        
        # Display underlying city variables if column is present
        city_col = [c for c in df.columns if 'city' in c]
        if city_col:
            unique_cities = df[city_col[0]].dropna().unique()
            st.write(f"**Megacities Tracked:** {len(unique_cities)}")
            st.caption(", ".join(list(unique_cities)[:10]) + "...")

    with col2:
        st.subheader("Raw Data Sample View")
        st.dataframe(df.head(10), use_container_width=True)

    # 2. Extract or Generate Model Pipelines
    st.divider()
    st.subheader("🤖 Predict Urban Particulate Matter ($PM_{2.5}$)")
    
    model_file = 'aqi_model.pkl'
    
    if not os.path.exists(model_file):
        with st.spinner("Model file not found locally. Initiating operational training pipeline..."):
            pipeline, features, target = train_aqi_model(df)
            st.toast("Model trained dynamically!", icon="✅")
    else:
        pipeline = joblib.load(model_file)
        features = pipeline.feature_names_
        target = pipeline.target_name_

    # 3. Dynamic User Inputs Generation
    st.write("Modify emission profiles below to evaluate simulated structural atmosphere effects:")
    
    input_data = {}
    input_cols = st.columns(len(features))
    
    for idx, feature in enumerate(features):
        with input_cols[idx]:
            # Calculate metrics bounds safely
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            
            # Numeric step slider allocation
            input_data[feature] = st.slider(
                label=f"Input {feature.upper()}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100.0
            )

    # 4. Process Inference Engines
    if st.button("Generate Inference Forecast", type="primary"):
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]
        
        # UI visualization output indicators
        st.markdown("---")
        st.markdown(f"### Predicted **{target.upper()}** Density Level:")
        
        if prediction < 35.5:
            st.success(f"🍃 Safe / Healthy: **{prediction:.2f} \mu g/m^3**")
        elif prediction < 55.5:
            st.warning(f"⚠️ Moderate Air Profile: **{prediction:.2f} \mu g/m^3**")
        else:
            st.error(f"🚨 Hazardous Air Overload: **{prediction:.2f} \mu g/m^3**")

except Exception as e:
    st.error(f"Execution Error Intercepted: {e}")
    st.info("Check that your system profile handles automated file tracking access rights correctly.")