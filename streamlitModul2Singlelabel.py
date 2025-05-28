import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import re
import string
import warnings
from sklearn.naive_bayes import MultinomialNB

# Initialize session state variables
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "label_column" not in st.session_state:
    st.session_state.label_column = None
if "evaluation_run" not in st.session_state:
    st.session_state.evaluation_run = False
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "X_test_tfidf" not in st.session_state:
    st.session_state.X_test_tfidf = None

# Set page configuration
st.set_page_config(page_title="Single-label Text Classification", layout="wide")

# Add title and description
st.title("Modul 2 - Single-label Text Classification")
st.markdown(
    "Single label klasifikasi teks menggunakan model Random Forest, SVM, dan Multinomial Naive Bayes."
)

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["Dataset Explorer", "Model Training", "Model Evaluation", "Prediction"],
)


# Define preprocessing function
@st.cache_data
def preprocess_text(text):
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text.lower()


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/train_preprocess.csv")
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
        ["fuel", "machine", "part", "others", "price", "service"],
    )

    # Show label distribution
    st.subheader(f"{label_to_explore.capitalize()} Label Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    df[label_to_explore].value_counts().plot(kind="bar", ax=ax)
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
        sentiment_percent = df[label_to_explore].value_counts(normalize=True) * 100
        st.dataframe(sentiment_percent.round(2).astype(str) + "%")

# Page 2: Model Training
elif page == "Model Training":
    st.header("Model Training")

    # Choose which label to train on
    label_column = st.selectbox(
        "Choose label to train:",
        ["fuel", "machine", "part", "others", "price", "service"],
    )

    # Model selection
    model_option = st.selectbox(
        "Select Model", ["Random Forest", "SVM", "Multinomial Naive Bayes"]
    )

    # Vectorization parameters
    st.subheader("Text Vectorization Parameters")
    max_features = st.slider(
        "Max Features", min_value=1000, max_value=10000, value=5000, step=1000
    )

    # Training parameters
    test_size = st.slider(
        "Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05
    )

    # Model-specific parameters
    if model_option == "Random Forest":
        n_estimators = st.slider(
            "Number of Trees", min_value=10, max_value=200, value=100, step=10
        )
    elif model_option == "SVM":
        C = st.slider(
            "Regularization Parameter (C)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
        )
    elif model_option == "Multinomial Naive Bayes":
        alpha = st.slider(
            "Smoothing Parameter (alpha)",
            min_value=0.01,
            max_value=1.0,
            value=1.0,
            step=0.01,
        )

    # Processing
    if st.button("Train Model"):
        st.info("Training in progress...")

        # Split data
        X = df["sentence"]
        y = df[label_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train model
        if model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
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

        # Save test data for evaluation
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_test_tfidf = X_test_tfidf

        # Quick validation (brief insights)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        st.success("Training complete!")
        st.subheader("Quick Validation")
        st.write(f"Validation Accuracy: {accuracy:.4f}")

        # Display a few example predictions
        st.info(
            "For detailed evaluation metrics, please go to the 'Model Evaluation' page."
        )

# Page 3: Model Evaluation
elif page == "Model Evaluation":
    st.header("Model Evaluation")

    # Check if there's a trained model
    if st.session_state.trained_model is None:
        st.warning(
            "No trained model found. Please train a model in the 'Model Training' page first."
        )
    else:
        st.success(
            f"Using trained {st.session_state.model_name} model for {st.session_state.label_column} classification"
        )

        # Initialize session state for CM view if not exists
        if "cm_view" not in st.session_state:
            st.session_state.cm_view = "Default"

        # Data source selection
        data_source = st.radio(
            "Select evaluation data source",
            [
                "Use test split from training",
                "Upload new test data",
                "Use random samples from dataset",
            ],
        )

        # Get model and resources
        model = st.session_state.trained_model
        vectorizer = st.session_state.vectorizer
        label_column = st.session_state.label_column

        # Based on selection, prepare test data
        test_data_ready = False

        if data_source == "Use test split from training":
            if "X_test" in st.session_state and "y_test" in st.session_state:
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                X_test_tfidf = st.session_state.X_test_tfidf
                test_data_ready = True
                st.info(f"Using test data from training split: {len(X_test)} samples")
            else:
                st.error(
                    "Test data from training not available. Please retrain the model or choose another option."
                )

        elif data_source == "Upload new test data":
            uploaded_file = st.file_uploader(
                "Upload test data (CSV format)", type=["csv"], key="eval_data_upload"
            )

            if uploaded_file is not None:
                # Load uploaded data
                try:
                    test_df = pd.read_csv(uploaded_file)

                    # Verify required columns exist
                    if (
                        label_column not in test_df.columns
                        or "sentence" not in test_df.columns
                    ):
                        st.error(
                            f"Uploaded file must contain 'sentence' and '{label_column}' columns."
                        )
                    else:
                        # Preprocess the text
                        test_df["sentence"] = test_df["sentence"].apply(preprocess_text)

                        # Extract features and labels
                        X_test = test_df["sentence"]
                        y_test = test_df[label_column]

                        # Vectorize text
                        X_test_tfidf = vectorizer.transform(X_test)
                        test_data_ready = True
                        st.info(f"Using uploaded test data: {len(X_test)} samples")

                except Exception as e:
                    st.error(f"Error processing uploaded file: {e}")

        else:  # Use random samples
            # Take a random sample from the dataset
            sample_size = st.slider(
                "Number of random samples",
                min_value=100,
                max_value=len(df),
                value=min(500, len(df)),
            )

            # Add a seed option for reproducibility
            seed = st.number_input(
                "Random seed (for reproducibility)",
                value=42,
                min_value=0,
                max_value=9999,
            )

            # Create sample
            np.random.seed(seed)
            random_indices = np.random.choice(len(df), size=sample_size, replace=False)
            test_df = df.iloc[random_indices]

            # Extract features and labels
            X_test = test_df["sentence"]
            y_test = test_df[label_column]

            # Vectorize text
            X_test_tfidf = vectorizer.transform(X_test)
            test_data_ready = True
            st.info(f"Using {sample_size} random samples from the dataset")

        # Evaluate button
        if test_data_ready:
            col1, col2 = st.columns([1, 3])

            with col1:
                evaluate_button = st.button(
                    "Run Evaluation", type="primary", use_container_width=True
                )

            with col2:
                if st.session_state.evaluation_run:
                    st.success("Evaluation complete! View results below")
                else:
                    st.info("Click 'Run Evaluation' to analyze model performance")

            # Process evaluation when button is clicked
            if evaluate_button:
                with st.spinner("Evaluating model performance..."):
                    # Make predictions
                    y_pred = model.predict(X_test_tfidf)

                    # Store prediction results in session state
                    st.session_state.eval_y_pred = y_pred
                    st.session_state.eval_y_test = y_test
                    st.session_state.eval_X_test = X_test

                    # Get unique classes
                    classes = sorted(y_test.unique())
                    st.session_state.eval_classes = classes

                    # Calculate confusion matrix
                    cm = confusion_matrix(y_test, y_pred, labels=classes)
                    st.session_state.eval_confusion_matrix = cm

                    # Calculate overall metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    )
                    recall = recall_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    )
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    st.session_state.eval_accuracy = accuracy
                    st.session_state.eval_precision = precision
                    st.session_state.eval_recall = recall
                    st.session_state.eval_f1 = f1

                    # Get detailed classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.session_state.eval_report = report_df

                    # Calculate per-class metrics
                    class_metrics = {}
                    for cls in classes:
                        tp = np.sum((y_test == cls) & (y_pred == cls))
                        fp = np.sum((y_test != cls) & (y_pred == cls))
                        fn = np.sum((y_test == cls) & (y_pred != cls))

                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = (
                            2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0
                            else 0
                        )

                        support = np.sum(y_test == cls)

                        class_metrics[cls] = {
                            "precision": precision,
                            "recall": recall,
                            "f1-score": f1,
                            "support": support,
                        }

                    st.session_state.eval_class_metrics = class_metrics

                    # If model has predict_proba, get probabilities
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test_tfidf)
                        st.session_state.eval_proba = proba

                    # Mark evaluation as completed
                    st.session_state.evaluation_run = True

                    # Force a rerun to show results
                    st.rerun()

            # Display results if evaluation has been run
            if st.session_state.evaluation_run:
                st.header("Evaluation Results")

                # High-level metrics with gauges/progress bars
                st.subheader("Overall Performance Metrics")

                # Create 4 metrics in one row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Accuracy", f"{st.session_state.eval_accuracy:.4f}")
                    st.progress(st.session_state.eval_accuracy)

                with col2:
                    st.metric("Precision", f"{st.session_state.eval_precision:.4f}")
                    st.progress(st.session_state.eval_precision)

                with col3:
                    st.metric("Recall", f"{st.session_state.eval_recall:.4f}")
                    st.progress(st.session_state.eval_recall)

                with col4:
                    st.metric("F1 Score", f"{st.session_state.eval_f1:.4f}")
                    st.progress(st.session_state.eval_f1)

                # Use tabs for different visualizations
                tabs = st.tabs(
                    ["Classification Report", "Confusion Matrix", "Sample Predictions"]
                )

                # Tab 1: Classification Report
                with tabs[0]:
                    st.subheader("Detailed Classification Report")

                    # Get the metrics DataFrame
                    report_df = st.session_state.eval_report

                    # Add color formatting function
                    def color_scale(val):
                        # Colors optimized for dark mode backgrounds
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            if val > 0.9:
                                return "background-color: #22543d; color: #e6ffe6;"  # Deep green bg, light text
                            elif val > 0.7:
                                return "background-color: #2a4365; color: #e0eaff;"  # Deep blue bg, light text
                            elif val > 0.5:
                                return "background-color: #744210; color: #ffe6cc;"  # Deep orange bg, light text
                            else:
                                return "background-color: #742a2a; color: #ffeaea;"  # Deep red bg, light text
                        return ""

                    # Display the DataFrame with formatting
                    st.dataframe(
                        report_df.style.format(
                            {
                                "precision": "{:.4f}",
                                "recall": "{:.4f}",
                                "f1-score": "{:.4f}",
                                "support": "{:.0f}",
                            }
                        ).applymap(
                            color_scale, subset=["precision", "recall", "f1-score"]
                        )
                    )

                    # Show per-class metrics in bar chart
                    st.subheader("F1 Score by Class")

                    classes = st.session_state.eval_classes
                    class_metrics = st.session_state.eval_class_metrics

                    # Extract F1 scores
                    f1_scores = [class_metrics[cls]["f1-score"] for cls in classes]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(classes, f1_scores, color="skyblue")

                    # Add value labels on top of bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{height:.4f}",
                            ha="center",
                            va="bottom",
                            rotation=0,
                        )

                    plt.xlabel("Class")
                    plt.ylabel("F1 Score")
                    plt.title("F1 Score by Class")
                    plt.ylim(0, 1.1)
                    plt.grid(axis="y", linestyle="--", alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # Tab 2: Confusion Matrix
                with tabs[1]:
                    st.subheader("Confusion Matrix")

                    # Get confusion matrix and classes
                    cm = st.session_state.eval_confusion_matrix
                    classes = st.session_state.eval_classes

                    # Create confusion matrix plot with annotations
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=classes,
                        yticklabels=classes,
                        ax=ax,
                    )
                    plt.title(f"Confusion Matrix for {label_column}")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Add normalized confusion matrix
                    st.subheader("Normalized Confusion Matrix")

                    # Normalize the confusion matrix
                    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0

                    # Create normalized confusion matrix plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        cm_norm,
                        annot=True,
                        fmt=".2f",
                        cmap="Blues",
                        xticklabels=classes,
                        yticklabels=classes,
                        ax=ax,
                    )
                    plt.title(f"Normalized Confusion Matrix for {label_column}")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # Tab 3: Sample Predictions
                with tabs[2]:
                    st.subheader("Sample Predictions")

                    # Get test data and predictions
                    X_test = st.session_state.eval_X_test
                    y_test = st.session_state.eval_y_test
                    y_pred = st.session_state.eval_y_pred

                    # Add filter options
                    filter_option = st.radio(
                        "Filter predictions",
                        ["Show all", "Show only correct", "Show only incorrect"],
                        index=0,
                    )

                    # Add a slider to control number of samples shown
                    num_samples = st.slider(
                        "Number of predictions to show",
                        min_value=3,
                        max_value=min(20, len(X_test)),
                        value=10,
                    )

                    # Sample indices for predictions
                    all_indices = np.arange(len(X_test))

                    # Determine correct/incorrect predictions
                    if filter_option != "Show all":
                        correct_predictions = []
                        for idx in all_indices:
                            is_correct = y_test.iloc[idx] == y_pred[idx]
                            if (
                                is_correct and filter_option == "Show only correct"
                            ) or (
                                not is_correct
                                and filter_option == "Show only incorrect"
                            ):
                                correct_predictions.append(idx)

                        if not correct_predictions:
                            st.warning(
                                f"No {filter_option.split()[-1]} predictions found."
                            )
                            sample_indices = []
                        else:
                            sample_indices = np.random.choice(
                                correct_predictions,
                                size=min(num_samples, len(correct_predictions)),
                                replace=False,
                            )
                    else:
                        sample_indices = np.random.choice(
                            all_indices,
                            size=min(num_samples, len(all_indices)),
                            replace=False,
                        )

                    # Display samples in expandable sections
                    for i, idx in enumerate(sample_indices):
                        with st.expander(f"Example {i + 1}", expanded=i == 0):
                            text = (
                                X_test.iloc[idx]
                                if hasattr(X_test, "iloc")
                                else X_test[idx]
                            )
                            true_label = (
                                y_test.iloc[idx]
                                if hasattr(y_test, "iloc")
                                else y_test[idx]
                            )
                            pred_label = y_pred[idx]

                            st.markdown(f"**Text:** {text}")

                            col1, col2 = st.columns(2)

                            # Show true label with appropriate color
                            with col1:
                                color = "gray"
                                if true_label == "positive":
                                    color = "green"
                                elif true_label == "negative":
                                    color = "red"
                                st.markdown(
                                    f"**True Label:** <span style='color:{color};'>{true_label}</span>",
                                    unsafe_allow_html=True,
                                )

                            # Show predicted label with appropriate color
                            with col2:
                                color = "gray"
                                if pred_label == "positive":
                                    color = "green"
                                elif pred_label == "negative":
                                    color = "red"
                                st.markdown(
                                    f"**Predicted Label:** <span style='color:{color};'>{pred_label}</span>",
                                    unsafe_allow_html=True,
                                )

                            # Show if prediction is correct
                            is_correct = true_label == pred_label
                            st.markdown(
                                f"**Prediction Status:** {'✅ Correct' if is_correct else '❌ Incorrect'}"
                            )

                            # Show probabilities if available
                            if hasattr(st.session_state, "eval_proba"):
                                st.subheader("Prediction Probabilities")
                                proba = st.session_state.eval_proba[idx]
                                classes = model.classes_

                                # Create DataFrame for probabilities
                                proba_df = pd.DataFrame(
                                    {"Class": classes, "Probability": proba}
                                )
                                proba_df = proba_df.sort_values(
                                    "Probability", ascending=False
                                )

                                # Create bar chart for probabilities
                                fig, ax = plt.subplots(figsize=(6, 3))
                                bars = ax.bar(
                                    proba_df["Class"],
                                    proba_df["Probability"],
                                    color=[
                                        "green"
                                        if c == "positive"
                                        else "red"
                                        if c == "negative"
                                        else "gray"
                                        for c in proba_df["Class"]
                                    ],
                                )

                                # Add percentage labels on top of bars
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.text(
                                        bar.get_x() + bar.get_width() / 2.0,
                                        height,
                                        f"{height:.2%}",
                                        ha="center",
                                        va="bottom",
                                        rotation=0,
                                    )

                                plt.ylim(0, 1.0)
                                plt.ylabel("Probability")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                # Download evaluation report section
                st.subheader("Export Evaluation Results")

                # Columns for download options
                col1, col2 = st.columns(2)

                with col1:
                    # Create a report DataFrame for detailed results
                    report_df = pd.DataFrame()
                    report_df["Text"] = X_test.reset_index(drop=True)
                    report_df["True_Label"] = y_test.reset_index(drop=True)
                    report_df["Predicted_Label"] = y_pred
                    report_df["Correct"] = (
                        report_df["True_Label"] == report_df["Predicted_Label"]
                    )

                    # Convert to CSV for download
                    csv = report_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Detailed Predictions",
                        data=csv,
                        file_name="evaluation_detailed.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with col2:
                    # Create a summary report
                    summary_dict = {
                        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                        "Score": [
                            st.session_state.eval_accuracy,
                            st.session_state.eval_precision,
                            st.session_state.eval_recall,
                            st.session_state.eval_f1,
                        ],
                    }

                    # Add per-class F1 scores to summary
                    for cls in st.session_state.eval_classes:
                        summary_dict["Metric"].append(f"F1 - {cls}")
                        summary_dict["Score"].append(
                            st.session_state.eval_class_metrics[cls]["f1-score"]
                        )

                    # Convert to DataFrame
                    summary_df = pd.DataFrame(summary_dict)

                    # Convert to CSV for download
                    summary_csv = summary_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Summary Metrics",
                        data=summary_csv,
                        file_name="evaluation_summary.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

# Page 4: Prediction
elif page == "Prediction":
    st.header("Make Predictions")

    # Input text
    user_input = st.text_area(
        "tulis text bebas:", "Avanza bahan bakar nya boros banget"
    )

    # Check if we have trained model stored in session state
    if st.session_state.trained_model is None:
        st.warning(
            "No trained model found. Please train a model in the 'Model Training' page first."
        )
        model_status = st.empty()
    else:
        st.success(
            f"Using trained {st.session_state.model_name} model for {st.session_state.label_column} classification"
        )

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
                prob_df = pd.DataFrame(
                    {"Class": class_labels, "Probability": probabilities}
                )
                prob_df = prob_df.sort_values("Probability", ascending=False)
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
                f"<h3 style='color:{color};'>{prediction}</h3>", unsafe_allow_html=True
            )

            # Show probabilities if available
            if prob_df is not None:
                st.subheader("Prediction Probabilities")

                # Create bar chart for probabilities
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(
                    prob_df["Class"],
                    prob_df["Probability"],
                    color=[
                        "green"
                        if c == "positive"
                        else "red"
                        if c == "negative"
                        else "gray"
                        for c in prob_df["Class"]
                    ],
                )

                # Add percentage labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2%}",
                        ha="center",
                        va="bottom",
                        rotation=0,
                    )

                plt.title(f"Prediction Probabilities for {label_column}")
                plt.ylim(0, 1.0)
                plt.ylabel("Probability")
                plt.tight_layout()
                st.pyplot(fig)

        else:
            # Train a default model if none is available
            model_status.info(
                "No trained model found. Training a default model Random Forest...."
            )

            # Define label column
            label_column = "fuel"  # Default to fuel sentiment

            # Create training data
            y = df[label_column]

            # Split data
            X_train, _, y_train, _ = train_test_split(
                df["sentence"], y, test_size=0.2, random_state=42
            )

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
                f"<h3 style='color:{color};'>{prediction}</h3>", unsafe_allow_html=True
            )
