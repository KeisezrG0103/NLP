import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
if 'label_column' not in st.session_state:
    st.session_state.label_column = None

# Set page configuration
st.set_page_config(
    page_title="Single-label Text Classification", layout="wide")

# Add title and description
st.title("Modul 2 - Single-label Text Classification")
st.markdown(
    "Single label klasifikasi teks menggunakan model Random Forest, SVM, dan Multinomial Naive Bayes.")

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
    st.subheader("Data")
    st.dataframe(df.head(20))

    # Choose which label to explore
    label_to_explore = st.selectbox(
        "Choose label to explore:",
        ["fuel", "machine", "part", "others", "price", "service"]
    )

    # Show label distribution
    st.subheader(f"{label_to_explore.capitalize()} Label Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    df[label_to_explore].value_counts().plot(kind='bar', ax=ax)
    plt.title(f"Distribution of {label_to_explore}")
    plt.tight_layout()
    st.pyplot(fig)

    # Show some insights
    st.subheader("Label Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Distribution by sentiment")
        sentiment_counts = df[label_to_explore].value_counts()
        st.dataframe(sentiment_counts)

    with col2:
        st.write("Percentage distribution")
        sentiment_percent = df[label_to_explore].value_counts(
            normalize=True) * 100
        st.dataframe(sentiment_percent.round(2).astype(str) + '%')

# Page 2: Model Training
elif page == "Model Training":
    st.header("Model Training")

    # Choose which label to train on
    label_column = st.selectbox(
        "Choose label to train:",
        ["fuel", "machine", "part", "others", "price", "service"]
    )

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
        y = df[label_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train model
        if model_option == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42)
        elif model_option == "SVM":
            model = SVC(C=C, probability=True, random_state=42)
        else:
            model = MultinomialNB(alpha=alpha)

        model.fit(X_train_tfidf, y_train)

        # Save model, vectorizer, and parameters to session state
        st.session_state.trained_model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.model_name = model_option
        st.session_state.label_column = label_column

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
        comparison_df['Actual'] = y_test.reset_index(drop=True)
        comparison_df['Predicted'] = y_pred
        comparison_df['Match'] = comparison_df['Actual'] == comparison_df['Predicted']

        st.dataframe(comparison_df.head(10))

        # Confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = sorted(y_test.unique())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        plt.title(f'Confusion Matrix for {label_column}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)

        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

# Page 3: Prediction
elif page == "Prediction":
    st.header("Make Predictions")

    # Input text
    user_input = st.text_area("tulis text bebas:",
                              "Avanza bahan bakar nya boros banget")

    # Check if we have trained model stored in session state
    if st.session_state.trained_model is None:
        st.warning(
            "No trained model found. Please train a model in the 'Model Training' page first.")
        model_status = st.empty()
    else:
        st.success(
            f"Using trained {st.session_state.model_name} model for {st.session_state.label_column} classification")

    if st.button("Predict"):
        st.info("Predicting...")

        # Preprocess input
        preprocessed_input = preprocess_text(user_input)

        # Check if we have trained model stored
        if st.session_state.trained_model is not None:
            # Use the trained model from session state
            model = st.session_state.trained_model
            vectorizer = st.session_state.vectorizer
            label_column = st.session_state.label_column

            # Transform input using stored vectorizer
            input_tfidf = vectorizer.transform([preprocessed_input])

            # Make prediction
            prediction = model.predict(input_tfidf)[0]

            # Get probability if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_tfidf)[0]
                class_labels = model.classes_
                prob_df = pd.DataFrame({
                    'Class': class_labels,
                    'Probability': probabilities
                })
                prob_df = prob_df.sort_values('Probability', ascending=False)
            else:
                prob_df = None

            # Display results
            st.success("Prediction complete!")

            # Show the input text
            st.subheader("Input Text")
            st.write(user_input)

            st.subheader("Preprocessed Text")
            st.write(preprocessed_input)

            # Show the prediction with nice formatting
            st.subheader(f"Predicted {label_column.capitalize()} Sentiment")

            # Set color based on sentiment
            color = "gray"
            if prediction == "positive":
                color = "green"
            elif prediction == "negative":
                color = "red"

            st.markdown(
                f"<h3 style='color:{color};'>{prediction}</h3>", unsafe_allow_html=True)

            # Show probabilities if available
            if prob_df is not None:
                st.subheader("Prediction Probabilities")

                # Create bar chart for probabilities
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(prob_df['Class'], prob_df['Probability'],
                              color=['green' if c == 'positive' else 'red' if c == 'negative' else 'gray' for c in prob_df['Class']])

                # Add percentage labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2%}',
                            ha='center', va='bottom', rotation=0)

                plt.title(f"Prediction Probabilities for {label_column}")
                plt.ylim(0, 1.0)
                plt.ylabel("Probability")
                plt.tight_layout()
                st.pyplot(fig)

        else:
            # Train a default model if none is available
            model_status.info(
                "No trained model found. Training a default model Random Forest....")

            # Define label column
            label_column = "fuel"  # Default to fuel sentiment

            # Create training data
            y = df[label_column]

            # Split data
            X_train, _, y_train, _ = train_test_split(
                df['sentence'], y, test_size=0.2, random_state=42)

            # Vectorize properly
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            input_tfidf = vectorizer.transform([preprocessed_input])

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_tfidf, y_train)

            # Save to session state for future use
            st.session_state.trained_model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.model_name = "Random Forest (default)"
            st.session_state.label_column = label_column

            # Make prediction
            prediction = model.predict(input_tfidf)[0]

            # Display results
            st.success("Default model trained and prediction complete!")

            # Show the input text
            st.subheader("Input Text")
            st.write(user_input)

            st.subheader("Preprocessed Text")
            st.write(preprocessed_input)

            # Show the prediction with nice formatting
            st.subheader(f"Predicted {label_column.capitalize()} Sentiment")

            # Set color based on sentiment
            color = "gray"
            if prediction == "positive":
                color = "green"
            elif prediction == "negative":
                color = "red"

            st.markdown(
                f"<h3 style='color:{color};'>{prediction}</h3>", unsafe_allow_html=True)
