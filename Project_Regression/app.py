from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("random_forest_model.pkl")

# Load the list of feature columns saved during training
feature_columns = joblib.load("feature_columns.pkl")  # Ensure this file exists

# Extract lists for one-hot encoding info (if needed for other uses)
locations_list = [col.replace('location_', '') for col in feature_columns if col.startswith('location_')]
area_types_list = ['Built_up_Area', 'Carpet_Area', 'Plot_Area', 'Super_built_up_Area']
balcony_values = [0.0, 1.0, 2.0, 3.0]

def preprocess_input(raw_input):
    # Initialize all features with 0
    data = dict.fromkeys(feature_columns, 0)
    
    # Assign numeric features
    data['bath'] = raw_input['bath']
    data['total_sqft'] = raw_input['total_sqft']
    data['bhk'] = raw_input['size']
    data['bath_per_size'] = raw_input['bath'] / raw_input['size'] if raw_input['size'] else 0
    
    # One-hot encode balcony count feature
    balcony_col = f'balcony_{float(raw_input["balcony"])}'
    if balcony_col in feature_columns:
        data[balcony_col] = 1
    
    # One-hot encode location feature
    loc_col = 'location_' + raw_input['location']
    if loc_col in feature_columns:
        data[loc_col] = 1
    else:
        # Use 'Other' if location unknown
        data['location_Other'] = 1
    
    # One-hot encode area_type feature
    area = raw_input['area_type']
    if area in feature_columns:
        data[area] = 1
    
    # Return as single-row dataframe suitable for model input
    return pd.DataFrame([data])

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        size = request.form.get('size')
        total_sqft = request.form.get('total_sqft')
        bath = request.form.get('bath')
        balcony = request.form.get('balcony')
        area_type = request.form.get('area_type')
        location = request.form.get('location')
        
        # Validate all inputs present
        if not all([size, total_sqft, bath, balcony, area_type, location]):
            return render_template('index.html', error="Please select all fields properly.")
        
        # Convert numeric fields to floats
        size = float(size)
        total_sqft = float(total_sqft)
        bath = float(bath)
        balcony = float(balcony)
        
        # Prepare raw input dictionary
        raw_input = {
            'size': size,
            'total_sqft': total_sqft,
            'bath': bath,
            'balcony': balcony,
            'area_type': area_type,
            'location': location
        }
        
        # Preprocess input for model
        input_df = preprocess_input(raw_input)
        
        # Predict price
        prediction = model.predict(input_df)[0]
        
        # Render result
        return render_template('index.html', prediction=round(prediction, 2))
    
    except Exception as e:
        # Handle any error and show message
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
