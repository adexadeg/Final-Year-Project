import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the trained model from 'model.pkl'
with open(r'C:\Users\Adegoke\Desktop\food-recommender-system\my-project\model.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Load the preprocessed dataset
file_path = r'C:\Users\Adegoke\Desktop\food-recommender-system\my-project\extract_data.csv'
extracted_data = pd.read_csv(file_path)

# Maximum nutritional values
max_Calories = 2000
max_daily_fat = 100
max_daily_Saturatedfat = 13
max_daily_Cholesterol = 300
max_daily_Sodium = 2300
max_daily_Carbohydrate = 325
max_daily_Fiber = 40
max_daily_Sugar = 40
max_daily_Protein = 200
max_list = [max_Calories, max_daily_fat, max_daily_Saturatedfat, max_daily_Cholesterol,
            max_daily_Sodium, max_daily_Carbohydrate, max_daily_Fiber, max_daily_Sugar, max_daily_Protein]

# Remove the unnecessary columns (not needed for modeling)
features = extracted_data[['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
                          'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']]

# Initialize the StandardScaler and fit it with the dataset
scaler = StandardScaler()
scaler.fit(features)

# Function to recommend recipes
def recommend_recipe(input_data, extracted_data, pipeline):
    input_data = scaler.transform(input_data)
    recommended_indices = pipeline.transform(input_data)[0]
    recommendations = extracted_data.iloc[recommended_indices]
    return recommendations

# Streamlit UI
st.title("Food Recommendation System")
st.write("Enter the nutritional values for your food to get recommendations.")

# Input form
input_data = st.form("Enter Nutritional Values for Your Recipe")
calories = input_data.number_input("Calories", min_value=extracted_data["Calories"].min(), max_value=extracted_data["Calories"].max())
fat_content = input_data.number_input("Fat Content (g)", min_value=extracted_data["FatContent"].min(), max_value=extracted_data["FatContent"].max())
saturated_fat = input_data.number_input("Saturated Fat Content (g)", min_value=extracted_data["SaturatedFatContent"].min(), max_value=extracted_data["SaturatedFatContent"].max())
cholesterol = input_data.number_input("Cholesterol Content (mg)", min_value=extracted_data["CholesterolContent"].min(), max_value=extracted_data["CholesterolContent"].max())
sodium = input_data.number_input("Sodium Content (mg)", min_value=extracted_data["SodiumContent"].min(), max_value=extracted_data["SodiumContent"].max())
carbohydrates = input_data.number_input("Carbohydrate Content (g)", min_value=extracted_data["CarbohydrateContent"].min(), max_value=extracted_data["CarbohydrateContent"].max())
fiber = input_data.number_input("Fiber Content (g)", min_value=extracted_data["FiberContent"].min(), max_value=extracted_data["FiberContent"].max())
sugar = input_data.number_input("Sugar Content (g)", min_value=extracted_data["SugarContent"].min(), max_value=extracted_data["SugarContent"].max())
protein = input_data.number_input("Protein Content (g)", min_value=extracted_data["ProteinContent"].min(), max_value=extracted_data["ProteinContent"].max())
submitted = input_data.form_submit_button("Get Recommendations")

if submitted:
    input_recipe = np.array([[calories, fat_content, saturated_fat, cholesterol, sodium, carbohydrates, fiber, sugar, protein]])
    
    recommendations = recommend_recipe(input_recipe, extracted_data, pipeline)

    st.header("Recommended Food and Recipes")
    st.dataframe(recommendations)

st.write("Note: Make sure to enter nutritional values within the specified limits.")

# Display the histogram
st.subheader("Histogram of Calories")
fig, ax = plt.subplots()
ax.hist(extracted_data['Calories'], bins=10, edgecolor='black')
st.pyplot(fig)
# Display the correlation matrix
st.subheader("Correlation Matrix")
corr_matrix = extracted_data.iloc[:, 6:15].corr(numeric_only=True)
st.write(corr_matrix)
