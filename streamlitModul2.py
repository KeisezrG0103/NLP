import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix
import re
import string
import warnings
from sklearn.naive_bayes import MultinomialNB

# Initialize session state variables to store the model and vectorizer
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'label_columns' not in st.session_state:
    st.session_state.label_columns = None

# Set page configuration
st.set_page_config(page_title="Multi-label Text Classification", layout="wide")

# Add title and description
st.title("Automotive Reviews Multi-label Text Classification")
st.markdown("multi label modul 2")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a page", ["Dataset Explorer", "Model Training", "Prediction"])

# Define preprocessing function


@st.cache_data
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text.lower()

# Load data


@st.cache_data
def load_data():
    df = pd.read_csv('data/train_preprocess.csv')
    return df


df = load_data()

# Page 1: Dataset Explorer
if page == "Dataset Explorer":
    st.header("Dataset Explorer")

    # Show basic dataset info
    st.subheader("Dataset Overview")
    st.write(f"Number of samples: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1]}")

    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Show label distribution
    st.subheader("Label Distribution")
    cols = st.columns(3)

    for i, column in enumerate(['fuel', 'machine', 'part']):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(5, 3))
            df[column].value_counts().plot(kind='bar', ax=ax)
            plt.title(f"Distribution of {column}")
            plt.tight_layout()
            st.pyplot(fig)

# Page 2: Model Training
elif page == "Model Training":
    st.header("Model Training")

    # Model selection
    model_option = st.selectbox(
        "Select Model", ["Random Forest", "SVM", "Multinomial Naive Bayes"])

    # Vectorization parameters
    st.subheader("Text Vectorization Parameters")
    max_features = st.slider(
        "Max Features", min_value=1000, max_value=10000, value=5000, step=1000)

    # Training parameters
    test_size = st.slider("Test Size", min_value=0.1,
                          max_value=0.5, value=0.2, step=0.05)

    # Model-specific parameters
    if model_option == "Random Forest":
        n_estimators = st.slider(
            "Number of Trees", min_value=10, max_value=200, value=100, step=10)
    elif model_option == "SVM":
        C = st.slider("Regularization Parameter (C)",
                      min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    elif model_option == "Multinomial Naive Bayes":
        alpha = st.slider("Smoothing Parameter (alpha)",
                          min_value=0.01, max_value=1.0, value=1.0, step=0.01)

    # Processing
    if st.button("Train Model"):
        st.info("Training in progress...")

        # Split data
        X = df['sentence']

        # Define label columns
        label_columns = ['fuel_negative', 'fuel_neutral', 'fuel_positive',
                         'machine_negative', 'machine_neutral', 'machine_positive',
                         'part_negative', 'part_neutral', 'part_positive']

        # Create multilabel target
        y_multilabel = pd.DataFrame()
        for sentiment in ['fuel', 'machine', 'part']:
            for label in ['negative', 'neutral', 'positive']:
                col_name = f"{sentiment}_{label}"
                y_multilabel[col_name] = (df[sentiment] == label).astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multilabel, test_size=test_size, random_state=42)

        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train model
        if model_option == "Random Forest":
            classifier = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42)
        elif model_option == "SVM":
            classifier = SVC(C=C, probability=True, random_state=42)
        else:
            classifier = MultinomialNB(alpha=alpha)

        model = BinaryRelevance(classifier=classifier)
        model.fit(X_train_tfidf, y_train)

        # Save model, vectorizer, and parameters to session state
        st.session_state.trained_model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.model_name = model_option
        st.session_state.label_columns = label_columns

        # Make predictions
        y_pred = model.predict(X_test_tfidf)

        # Display results
        st.success("Training complete!")
        st.subheader("Model Performance")

        # Display accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Overall Accuracy: {accuracy:.4f}")

        # Side by side comparison
        st.subheader("Side-by-Side Comparison")

        comparison_df = pd.DataFrame()
        comparison_df['Text'] = X_test.reset_index(drop=True)

        # Add columns for actual and predicted values
        for label_col in label_columns[:3]:  # Just showing first 3 for clarity
            comparison_df[f'{label_col}_actual'] = y_test[label_col].reset_index(
                drop=True)
            comparison_df[f'{label_col}_predicted'] = y_pred.toarray()[
                :, label_columns.index(label_col)]
            comparison_df[f'{label_col}_match'] = comparison_df[f'{label_col}_actual'] == comparison_df[f'{label_col}_predicted']

        st.dataframe(comparison_df.head(10))

        # Confusion matrices
        st.subheader("Confusion Matrices")

        mcm = multilabel_confusion_matrix(y_test, y_pred.toarray())

        # Add option to select which confusion matrices to display
        # Store confusion matrices in session state
        if 'confusion_matrices' not in st.session_state:
            st.session_state.confusion_matrices = mcm
            st.session_state.mcm_labels = label_columns

        # Create 3 rows of 3 columns for all 9 matrices
        for row in range(3):
            cols = st.columns(3)

            for col in range(3):
                label_idx = row * 3 + col
                if label_idx < len(label_columns):
                    with cols[col]:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(mcm[label_idx], annot=True,
                                    fmt='d', cmap='Blues', ax=ax)
                        plt.title(
                            f'Confusion Matrix: {label_columns[label_idx]}')
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

# Page 3: Prediction
elif page == "Prediction":
    st.header("Make Predictions")

    # Input text
    user_input = st.text_area("tuliskan review (terserah apa saja):",
                              "Avanza bahan bakar nya boros banget")

    # Check if we have trained model stored in session state
    if st.session_state.trained_model is None:
        st.warning(
            "No trained model found. Please train a model in the 'Model Training' page first.")
        model_status = st.empty()
    else:
        st.success(f"Using trained {st.session_state.model_name} model")

    if st.button("Predict"):
        st.info("Predicting...")

        # Preprocess input
        preprocessed_input = preprocess_text(user_input)

        # Check if we have trained model stored
        if st.session_state.trained_model is not None:
            # Use the trained model from session state
            model = st.session_state.trained_model
            vectorizer = st.session_state.vectorizer
            label_columns = st.session_state.label_columns

            # Transform input using stored vectorizer
            input_tfidf = vectorizer.transform([preprocessed_input])
        else:
            # Train a default model if none is available
            model_status.info(
                "No trained model found. Training a default model Random Forest....")

            # Define label columns
            label_columns = ['fuel_negative', 'fuel_neutral', 'fuel_positive',
                             'machine_negative', 'machine_neutral', 'machine_positive',
                             'part_negative', 'part_neutral', 'part_positive']

            # Create multilabel target for training
            y_multilabel = pd.DataFrame()
            for sentiment in ['fuel', 'machine', 'part']:
                for label in ['negative', 'neutral', 'positive']:
                    col_name = f"{sentiment}_{label}"
                    y_multilabel[col_name] = (
                        df[sentiment] == label).astype(int)

            # Split data
            X_train, _, y_train, _ = train_test_split(
                df['sentence'], y_multilabel, test_size=0.2, random_state=42)

            # Vectorize properly
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            input_tfidf = vectorizer.transform([preprocessed_input])

            # Train model
            model = BinaryRelevance(classifier=RandomForestClassifier(
                n_estimators=100, random_state=42))
            model.fit(X_train_tfidf, y_train)

            # Save to session state for future use
            st.session_state.trained_model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.model_name = "Random Forest (default)"
            st.session_state.label_columns = label_columns

        # Make prediction
        prediction = model.predict(input_tfidf)

        # Display results
        st.success("Prediction complete!")

        # Show the input text
        st.subheader("Input Text")
        st.write(user_input)

        st.subheader("Preprocessed Text")
        st.write(preprocessed_input)

        # Show the predicted labels (fixed format)
        st.subheader("Predicted Labels")

    # Create nicer display of predictions
        results = []
        has_predictions = False

        for i, label in enumerate(label_columns):
            if prediction.toarray()[0, i] == 1:
                results.append(label)
                has_predictions = True

        if has_predictions:
            for label in results:
                st.write(f"- {label}")

            # Display visual representation by category
            st.subheader("Prediction Summary")

            # Group by category
            fuel_preds = [col for col in results if col.startswith('fuel_')]
            machine_preds = [
                col for col in results if col.startswith('machine_')]
            part_preds = [col for col in results if col.startswith('part_')]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Fuel sentiment:**")
                if fuel_preds:
                    for pred in fuel_preds:
                        st.markdown(
                            f"<span style='color: red;'>- {pred.replace('fuel_', '')}</span>",
                            unsafe_allow_html=True)
                else:
                    st.write("No prediction")

            with col2:
                st.write("**Machine sentiment:**")
                if machine_preds:
                    for pred in machine_preds:
                        st.markdown(
                            f"<span style='color: green;'>- {pred.replace('machine_', '')}</span>",
                            unsafe_allow_html=True)
                else:
                    st.write("No prediction")

            with col3:
                st.write("**Part sentiment:**")
                if part_preds:
                    for pred in part_preds:
                        st.markdown(
                            f"<span style='color: blue;'>- {pred.replace('part_', '')}</span>",
                            unsafe_allow_html=True)
                else:
                    st.write("No prediction")
        else:
            st.write("No labels predicted.")
