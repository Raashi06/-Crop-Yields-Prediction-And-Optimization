import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Load Models (cached)
@st.cache_resource
def load_preprocessors():
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    encoder.categories_ = [
        ['West', 'South', 'North', 'East'],
        ['Sandy', 'Clay', 'Loam', 'Silt'],
        ['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize'],
        [False, True],
        [False, True],
        ['Cloudy', 'Sunny', 'Rainy', 'Windy']
    ]
    return scaler, encoder

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('gru_crop_yield_model.keras')
    return model

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Prediction'
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Sidebar Navigation

st.sidebar.markdown("## 🥸 Navigation")
st.sidebar.markdown("---")

pages = ['About Project', 'Prediction', 'Model Evaluation']

for page in pages:
    if st.sidebar.button(
        page,
        key=f"nav_{page}",
        use_container_width=True,
        type="primary" if st.session_state.current_page == page else "secondary"
    ):
        st.session_state.current_page = page
        st.rerun()

st.sidebar.markdown("---")


# PAGE: About Project
if st.session_state.current_page == 'About Project':
    st.markdown('<div class="main-header"><h1>🌾 Crop Yield Prediction System</h1></div>', unsafe_allow_html=True)
    
    st.markdown("## 📖 About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This intelligent crop yield prediction system uses **deep learning** to forecast agricultural yields 
        based on various environmental and agricultural factors.
        
        ### 🔬 Technology Stack
        - **Model**: GRU (Gated Recurrent Unit) Neural Network
        - **Framework**: TensorFlow/Keras
        - **Data Processing**: Scikit-learn, Pandas, NumPy
        - **Interface**: Streamlit
        
        ### 📊 Input Features
        - **Environmental**: Region, Soil Type, Weather Condition
        - **Climate**: Rainfall, Temperature
        - **Agricultural**: Crop Type, Fertilizer, Irrigation
        - **Timeline**: Days to Harvest
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Key Features
        - ✅ Multi-step prediction wizard
        - ✅ Real-time yield forecasting
        - ✅ 91.3% prediction accuracy
        - ✅ Support for 6 major crops
        - ✅ Multiple soil and weather conditions
        
        ### 🌾 Supported Crops
        - Cotton
        - Rice
        - Barley
        - Soybean
        - Wheat
        - Maize
        
        ### 🎯 Use Cases
        - Farm planning and optimization
        - Resource allocation
        - Risk assessment
        - Harvest forecasting
        """)
    
    st.markdown("---")
    st.info("👉 Navigate to **Prediction** to start forecasting yields or **Model Evaluation** to view performance metrics.")


# PAGE: Prediction
elif st.session_state.current_page == 'Prediction':
    st.markdown('<div class="main-header"><h1>🌾 Crop Yield Prediction</h1></div>', unsafe_allow_html=True)
    
    # Load models
    scaler, encoder = load_preprocessors()
    model = load_model()
    
    # Progress bar
    progress = (st.session_state.step - 1) / 4
    st.progress(progress, text=f"Step {st.session_state.step} of 5")
    
    # Validation function
    def validate_step(step):
        required_fields = {
            1: ['region', 'soil_type', 'crop'],
            2: ['rainfall_mm', 'temperature_celsius'],
            3: ['fertilizer_used', 'irrigation_used'],
            4: ['weather_condition', 'days_to_harvest']
        }
        
        for field in required_fields.get(step, []):
            if field not in st.session_state.form_data:
                return False
        return True
    
    # Step 1: Location & Crop
    if st.session_state.step == 1:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("📍 Step 1: Location & Crop Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox('Region', ['West', 'South', 'North', 'East'], 
                                  key='region_input',
                                  index=['West', 'South', 'North', 'East'].index(st.session_state.form_data.get('region', 'West')))
            st.session_state.form_data['region'] = region
            
        with col2:
            soil_type = st.selectbox('Soil Type', ['Sandy', 'Clay', 'Loam', 'Silt'],
                                     key='soil_input',
                                     index=['Sandy', 'Clay', 'Loam', 'Silt'].index(st.session_state.form_data.get('soil_type', 'Sandy')))
            st.session_state.form_data['soil_type'] = soil_type
        
        crop = st.selectbox('Crop Type', ['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize'],
                            key='crop_input',
                            index=['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize'].index(st.session_state.form_data.get('crop', 'Cotton')))
        st.session_state.form_data['crop'] = crop
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Climate Data
    elif st.session_state.step == 2:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("🌡️ Step 2: Climate Conditions")
        
        col1, col2 = st.columns(2)
        with col1:
            rainfall = st.number_input('Rainfall (mm)', 0.0, 2000.0, 
                                       st.session_state.form_data.get('rainfall_mm', 750.0),
                                       key='rainfall_input')
            st.session_state.form_data['rainfall_mm'] = rainfall
            
        with col2:
            temperature = st.number_input('Temperature (°C)', 0.0, 50.0, 
                                          st.session_state.form_data.get('temperature_celsius', 25.0),
                                          key='temp_input')
            st.session_state.form_data['temperature_celsius'] = temperature
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Agricultural Practices
    elif st.session_state.step == 3:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("🚜 Step 3: Agricultural Practices")
        
        col1, col2 = st.columns(2)
        with col1:
            fertilizer = st.checkbox('Fertilizer Used', 
                                     value=st.session_state.form_data.get('fertilizer_used', False),
                                     key='fert_input')
            st.session_state.form_data['fertilizer_used'] = fertilizer
            
        with col2:
            irrigation = st.checkbox('Irrigation Used', 
                                     value=st.session_state.form_data.get('irrigation_used', False),
                                     key='irr_input')
            st.session_state.form_data['irrigation_used'] = irrigation
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Weather & Timeline
    elif st.session_state.step == 4:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("⏰ Step 4: Weather & Harvest Timeline")
        
        col1, col2 = st.columns(2)
        with col1:
            weather = st.selectbox('Weather Condition', ['Cloudy', 'Sunny', 'Rainy', 'Windy'],
                                   key='weather_input',
                                   index=['Cloudy', 'Sunny', 'Rainy', 'Windy'].index(st.session_state.form_data.get('weather_condition', 'Sunny')))
            st.session_state.form_data['weather_condition'] = weather
            
        with col2:
            days = st.number_input('Days to Harvest', 1, 365, 
                                   st.session_state.form_data.get('days_to_harvest', 100),
                                   key='days_input')
            st.session_state.form_data['days_to_harvest'] = days
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 5: Review & Predict
    elif st.session_state.step == 5:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("✅ Step 5: Review & Predict")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Input Summary")
            st.write(f"**Region:** {st.session_state.form_data['region']}")
            st.write(f"**Soil Type:** {st.session_state.form_data['soil_type']}")
            st.write(f"**Crop:** {st.session_state.form_data['crop']}")
            st.write(f"**Rainfall:** {st.session_state.form_data['rainfall_mm']} mm")
            st.write(f"**Temperature:** {st.session_state.form_data['temperature_celsius']}°C")
        
        with col2:
            st.markdown("#### 🚜 Practices & Timeline")
            st.write(f"**Fertilizer:** {'Yes' if st.session_state.form_data['fertilizer_used'] else 'No'}")
            st.write(f"**Irrigation:** {'Yes' if st.session_state.form_data['irrigation_used'] else 'No'}")
            st.write(f"**Weather:** {st.session_state.form_data['weather_condition']}")
            st.write(f"**Days to Harvest:** {st.session_state.form_data['days_to_harvest']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.prediction is not None:
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
    
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
                margin: 1rem 0;
            ">
                <h2 style="color: white; margin: 0; font-size: 1.2rem; font-weight: normal;">
                    🌱 Predicted Crop Yield
                </h2>
                <h1 style="color: white; margin: 1rem 0; font-size: 3.5rem; font-weight: bold;">
                    {st.session_state.prediction:.2f}
                </h1>
                <h3 style="color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 1.5rem; font-weight: normal;">
                    tons/hectare
                </h3>
            </div>
            """, unsafe_allow_html=True)

    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.session_state.step > 1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()
    
    with col2:
        if st.session_state.step == 5 and st.session_state.prediction is not None:
            if st.button("🔄 Start New Prediction", use_container_width=True):
                st.session_state.step = 1
                st.session_state.form_data = {}
                st.session_state.prediction = None
                st.rerun()
    
    with col3:
        if st.session_state.step < 5:
            if st.button("Next ➡️", use_container_width=True, type="primary"):
                if validate_step(st.session_state.step):
                    st.session_state.step += 1
                    st.rerun()
                else:
                    st.error("Please fill in all required fields!")
        elif st.session_state.step == 5 and st.session_state.prediction is None:
            if st.button("🎯 Calculate Yield", use_container_width=True, type="primary"):
                # Create prediction
                data = st.session_state.form_data
                new_df = pd.DataFrame({
                    'Region': [data['region']],
                    'Soil_Type': [data['soil_type']],
                    'Crop': [data['crop']],
                    'Rainfall_mm': [data['rainfall_mm']],
                    'Temperature_Celsius': [data['temperature_celsius']],
                    'Fertilizer_Used': [data['fertilizer_used']],
                    'Irrigation_Used': [data['irrigation_used']],
                    'Weather_Condition': [data['weather_condition']],
                    'Days_to_Harvest': [data['days_to_harvest']]
                })
                
                numerical = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
                categorical = ['Region', 'Soil_Type', 'Crop', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition']
                
                new_df[numerical] = scaler.transform(new_df[numerical])
                encoded = encoder.transform(new_df[categorical])
                
                final_input = np.concatenate([encoded, new_df[numerical].values], axis=1)
                final_input = final_input.reshape(1, 1, final_input.shape[1])
                
                pred = model.predict(final_input, verbose=0)
                st.session_state.prediction = float(pred[0][0])
                st.rerun()


# PAGE: Model Evaluation
elif st.session_state.current_page == 'Model Evaluation':
    st.markdown('<div class="main-header"><h1>📊 Model Evaluation</h1></div>', unsafe_allow_html=True)
    
    st.markdown("## 📌 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="R² Score",
            value="0.9132",
            delta="High Accuracy",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="RMSE",
            value="0.5018",
            delta="Low Error",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="MAE",
            value="0.4005",
            delta="Excellent",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    st.info("""
    **Model Performance Summary:**
    - **R² Score of 0.9132** indicates that the model explains 91.32% of the variance in crop yields
    - **Low RMSE (0.502)** shows excellent prediction accuracy with minimal error
    - **MAE of 0.400** confirms consistent prediction quality across all samples
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Predicted vs Actual Yields")
        try:
            scatter_img = Image.open("scatter_plot.png")
            st.image(scatter_img, use_container_width=True)  
        except:
            st.warning("⚠️ scatter_plot.png not found. Please upload it to the project folder.")
    
    with col2:
        st.markdown("### 📍 Prediction Error Distribution")
        try:
            hist_img = Image.open("error_histogram.png")
            st.image(hist_img, use_container_width=True)  
        except:
            st.warning("⚠️ error_histogram.png not found. Please upload it to the project folder.")
    
    st.markdown("---")


# Custom CSS 
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;

        /* --- FIX: Make header sticky --- */
        position: sticky;
        top: 0; /* Sticks to the top of its scrolling container */
        z-index: 10; /* Ensures it stays on top of other content */
        /* Add a shadow to show it's "lifted" */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .step-container {
        background: transparent; /* Changed from white to transparent */
        padding: 2rem;
        border-radius: 10px;
        box-shadow: none; /* Removed box shadow */
        margin-bottom: 2rem;
    }
    .stApp {
        background-color: #f0fdf4; /* Light green background */
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dcfce7 0%, #bbf7d0 100%); /* Made gradient darker */
    }

    /* --- FIX: Re-styled sidebar buttons to match screenshot --- */

    /* Style inactive sidebar buttons (like screenshot) */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #374151; /* Dark Gray */
        color: white;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #4B5563; /* Lighter Dark Gray */
        color: white;
    }

    /* Style active sidebar button (like screenshot) */
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #EF4444; /* Red */
        color: white;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #DC2626; /* Darker Red */
        color: white;
    }
</style>
""", unsafe_allow_html=True)