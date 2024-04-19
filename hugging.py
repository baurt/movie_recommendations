import pandas as pd
import streamlit as st

# Read the CSV file into a DataFrame
df = pd.read_csv('output.csv')
df.fillna('', inplace=True)


import pickle

# Load the vectorizer from the file
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np


# Example DataFrame with image links


# Function to display image from URL
def display_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))    
    st.image(img.resize((400, 500)), use_column_width=False, width=300)






# Define the recommend_movies function
def recommend_movies(user_input, df, vectorizer, tfidf_matrix, tfidf_matrix2, top_n=st.sidebar.slider('Select number of recommendations',min_value=1,max_value=100,value=10,step=1)):
    user_tfidf = vectorizer.transform([user_input])
    similarity_scores_desc = cosine_similarity(user_tfidf, tfidf_matrix)
    similarity_scores_name = cosine_similarity(user_tfidf, tfidf_matrix2)
    similarity_scores = st.sidebar.slider('Select plot weight',min_value=0.0,max_value=1.0,value=0.7,step=0.1) * similarity_scores_desc + st.sidebar.slider('Select name weight',min_value=0.0,max_value=1.0,value=0.3,step=0.1) * similarity_scores_name
    top_indices = similarity_scores.argsort(axis=1)[:, ::-1][:, :top_n]
    recommended_movies = df.iloc[top_indices.ravel()]['Name'].values
    return recommended_movies ,top_indices[0]

user_input = st.text_input("Enter some text:")

if __name__ == "__main__":

    user_input2 = "фильм про карликов которые несут кольцо к горе"

    # Load your vectorizer object
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Define your tfidf_matrix and tfidf_matrix2
    tfidf_matrix = vectorizer.transform(df['Description'])
    tfidf_matrix2 = vectorizer.transform(df['Name'])

    # Call the recommend_movies function
    recommended_movies,indices = recommend_movies(user_input, df, vectorizer, tfidf_matrix, tfidf_matrix2)





# Assuming 'indices' is a list of indices for the movies you want to display
for i in indices:
    col1, col2 = st.columns([1, 1])  # Create two columns with a ratio of 1:3

    # Display movie image in the left column
    with col1:
        
        display_image_from_url(df["Image"][i])
        st.write(f"Filmru: {df['film.ru'].values[i]} IMDb: {df['IMDb'].values[i]} User: {df['User_rating'].values[i]} ")
        st.write(f"{df["Genre"].values[i]}")
        st.write(f"{df["Country"].values[i]}")


    # Display movie information in the right column
    with col2:
        st.subheader(df["Name"].values[i])
        st.write(df["Description"].values[i])
        

            

