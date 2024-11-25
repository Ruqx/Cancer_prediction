import streamlit as st

# Top navigation bar
navigation = st.radio("Navigate to:", ["Page 1", "Page 2"], horizontal=True)

# Page 1 content
if navigation == "Page 1":
    import base64

    # Custom CSS to change text color
    text_color_css = """
    <style>
    h1, h2, h3, h4, h5, h6, p {
        color: #000000; 
    }
    </style>
    """

    # Apply the custom CSS
    st.markdown(text_color_css, unsafe_allow_html=True)


    # Function to load and encode the image in Base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Set the background using the Base64-encoded image
    def set_background(image_path):
        encoded_image = get_base64_image(image_path)
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

    # Path to your local image file
    image_path = "a3.jpg"  # Replace with the path to your image

    # Set the background
    set_background(image_path)

    st.title("Disease Presdiction System for Cancer using Tumor")
    st.markdown(
        """
        <p style="font-size:18px;">
        <h3>Benign Tumors</h3>
        
        <b>Non-cancerous:</b> Benign tumors do not contain cancer cells.
        <b>Growth:</b> They grow slowly and do not invade nearby tissues.
        <b>Spread:</b> They do not spread to other parts of the body (no metastasis).
        <b>Appearance:</b> Cells in benign tumors look similar to normal cells and have a regular shape.
        <b>Boundaries:</b> They usually have clear boundaries and are often encapsulated.
        <b>Treatment:</b> Often, they do not require treatment unless they cause symptoms by pressing on nearby organs or tissues.

        <h3>Malignant Tumors</h3>
        
        <b>Cancerous:</b> Malignant tumors contain cancer cells.
        <b>Growth:</b> They grow rapidly and can invade nearby tissues.
        <b>Spread:</b> They can spread to other parts of the body through the bloodstream or lymphatic system (metastasis).
        <b>Appearance:</b> Cells in malignant tumors are abnormal and vary in shape and size.
        <b>Boundaries:</b> They often have irregular boundaries and can infiltrate surrounding tissues.
        <b>Treatment:</b> They usually require aggressive treatment, including surgery, chemotherapy, and radiation therapy.
        </p>
        """,
        unsafe_allow_html=True,
    )
# Page 2 content
elif navigation == "Page 2":
        #importing required libraries
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    #reading the dataset
    data = pd.read_csv("Cancer_Data.csv")
    # Drop the 'Unnamed: 32' column and the 'id' column as they are unnecessary
    data_cleaned = data.drop(['Unnamed: 32', 'id'], axis=1)
    # Check if any columns have missing values
    missing_values = data_cleaned.isnull().sum()

    # Load and clean the data (assuming data_cleaned already exists)
    X = data_cleaned.drop('diagnosis', axis=1)
    y = data_cleaned['diagnosis']
    # Select two features for visualization (e.g., radius_mean and texture_mean)
    X_viz = data_cleaned[['radius_mean', 'texture_mean']]
    y_viz = y
    # Split the data into training and testing sets for visualization (80% train, 20% test)
    X_train_viz, X_test_viz, y_train_viz, y_test_viz = train_test_split(X_viz, y_viz, test_size=0.2, random_state=42)
    # Standardize the two features
    scaler = StandardScaler()
    X_train_viz_scaled = scaler.fit_transform(X_train_viz)
    X_test_viz_scaled = scaler.transform(X_test_viz)
    # Train logistic regression model on the selected features
    model_viz = LogisticRegression(max_iter=10000)
    model_viz.fit(X_train_viz_scaled, y_train_viz)
    # Predictions, accuracy, classification report, and confusion matrix
    y_pred_viz = model_viz.predict(X_test_viz_scaled)
    accuracy_viz = accuracy_score(y_test_viz, y_pred_viz)
    class_report_viz = classification_report(y_test_viz, y_pred_viz)
    conf_matrix_viz = confusion_matrix(y_test_viz, y_pred_viz)

    # Import necessary libraries
    import numpy as np

    # Function to take user input for new cancer data
    def predict_cancer(radius, texture):
        try:
            # Accept input for the two features used in this model
            radius_mean = radius
            texture_mean = texture

            # Create an array from the input
            new_data = np.array([[radius_mean, texture_mean]])

            # Scale the input data using the same scaler
            new_data_scaled = scaler.transform(new_data)

            # Make predictions using the trained model
            prediction = model_viz.predict(new_data_scaled)
            prediction_prob = model_viz.predict_proba(new_data_scaled)

            # Map prediction to diagnosis
            diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

            # Display the prediction
            print(f"\nPrediction: {diagnosis}")
            print(f"Prediction probabilities: {prediction_prob}")
            return diagnosis
        except Exception as e:
            print(f"Error in input or prediction: {e}")

    import streamlit as st
    import base64

    # Custom CSS to change text color
    text_color_css = """
    <style>
    h1, h2, h3, h4, h5, h6, p {
        color: #000000; 
    }
    </style>
    """

    # Apply the custom CSS
    st.markdown(text_color_css, unsafe_allow_html=True)


    # Function to load and encode the image in Base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Set the background using the Base64-encoded image
    def set_background(image_path):
        encoded_image = get_base64_image(image_path)
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

    # Path to your local image file
    image_path = "a3.jpg"  # Replace with the path to your image

    # Set the background
    set_background(image_path)

    st.title("Disease Presdiction System for Cancer using Tumor")


    radius = st.number_input("Enter the radius of the tumor")
    texture = st.number_input("Enter the texture of the tumor")
    a= predict_cancer(radius,texture)

    if st.button("Show Result"):
        st.header(f"The Tumor is {a}")


