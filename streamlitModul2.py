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
    multilabel_confusion_matrix,
)
import re
import string
import warnings
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import (
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset,
)

# Initialize session state variables to store the model and vectorizer
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "label_columns" not in st.session_state:
    st.session_state.label_columns = None

# Set page configuration
st.set_page_config(page_title="Multi-label Text Classification", layout="wide")

# Add title and description
st.title("Automotive Reviews Multi-label Text Classification")
st.markdown("multi label modul 2")

# Sidebar for navigation
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
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Show label distribution
    st.subheader("Label Distribution")
    cols = st.columns(3)

    for i, column in enumerate(["fuel", "machine", "part"]):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(5, 3))
            df[column].value_counts().plot(kind="bar", ax=ax)
            plt.title(f"Distribution of {column}")
            plt.tight_layout()
            st.pyplot(fig)

# Page 2: Model Training

elif page == "Model Training":
    st.header("Model Training")

    # Multi-label strategy selection
    multilabel_strategy = st.selectbox(
        "Select Multi-label Strategy",
        ["Binary Relevance", "Classifier Chain", "Label Powerset"],
    )

    # Explanation of each strategy
    if multilabel_strategy == "Binary Relevance":
        st.info(
            "Binary Relevance trains an independent classifier for each label. Fast but ignores label correlations."
        )
    elif multilabel_strategy == "Classifier Chain":
        st.info(
            "Classifier Chain creates a chain of classifiers, each considering predictions of previous classifiers. Captures label dependencies."
        )
    else:  # Label Powerset
        st.info(
            "Label Powerset transforms multi-label into multi-class by treating each label combination as a unique class. Better for capturing label interactions but may suffer with many labels."
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

        # Define label columns
        label_columns = [
            "fuel_negative",
            "fuel_neutral",
            "fuel_positive",
            "machine_negative",
            "machine_neutral",
            "machine_positive",
            "part_negative",
            "part_neutral",
            "part_positive",
        ]

        # Create multilabel target
        y_multilabel = pd.DataFrame()
        for sentiment in ["fuel", "machine", "part"]:
            for label in ["negative", "neutral", "positive"]:
                col_name = f"{sentiment}_{label}"
                y_multilabel[col_name] = (df[sentiment] == label).astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multilabel, test_size=test_size, random_state=42
        )

        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train model
        if model_option == "Random Forest":
            classifier = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42
            )
        elif model_option == "SVM":
            classifier = SVC(C=C, probability=True, random_state=42)
        else:
            classifier = MultinomialNB(alpha=alpha)

        # Create model based on selected multi-label strategy
        if multilabel_strategy == "Binary Relevance":
            model = BinaryRelevance(classifier=classifier)
        elif multilabel_strategy == "Classifier Chain":
            model = ClassifierChain(classifier=classifier)
        else:  # Label Powerset
            model = LabelPowerset(classifier=classifier)

        model.fit(X_train_tfidf, y_train)

        # Save model, vectorizer, and parameters to session state
        st.session_state.trained_model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.model_name = f"{multilabel_strategy} with {model_option}"
        st.session_state.label_columns = label_columns
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_test_tfidf = X_test_tfidf

        # Brief training insights
        st.success("Training complete!")

        # Quick validation results
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Validation Accuracy: {accuracy:.4f}")

        st.info(
            "For detailed evaluation metrics, please go to the 'Model Evaluation' page."
        )
        
        
elif page == "Model Evaluation":
    st.header("Model Evaluation")

    # Check if there's a trained model
    if st.session_state.trained_model is None:
        st.warning(
            "No trained model found. Please train a model in the 'Model Training' page first."
        )
    else:
        st.success(f"Using trained {st.session_state.model_name} model")

        # Initialize evaluation session state if not existing
        if "evaluation_run" not in st.session_state:
            st.session_state.evaluation_run = False
        if "cm_view" not in st.session_state:
            st.session_state.cm_view = "By aspect"

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
        label_columns = st.session_state.label_columns

        # Based on selection, prepare test data
        test_data_ready = False

        if data_source == "Use test split from training":
            if (
                "X_test" in st.session_state
                and "y_test" in st.session_state
                and "X_test_tfidf" in st.session_state
            ):
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
                    required_columns = ["sentence", "fuel", "machine", "part"]
                    if not all(col in test_df.columns for col in required_columns):
                        st.error(
                            "Uploaded file must contain 'sentence', 'fuel', 'machine', and 'part' columns."
                        )
                    else:
                        # Preprocess the text
                        test_df["sentence"] = test_df["sentence"].apply(preprocess_text)

                        # Create multilabel target
                        y_test = pd.DataFrame()
                        for sentiment in ["fuel", "machine", "part"]:
                            for label in ["negative", "neutral", "positive"]:
                                col_name = f"{sentiment}_{label}"
                                y_test[col_name] = (test_df[sentiment] == label).astype(
                                    int
                                )

                        # Vectorize text
                        X_test = test_df["sentence"]
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

            # Create multilabel target
            y_test = pd.DataFrame()
            for sentiment in ["fuel", "machine", "part"]:
                for label in ["negative", "neutral", "positive"]:
                    col_name = f"{sentiment}_{label}"
                    y_test[col_name] = (test_df[sentiment] == label).astype(int)

            # Vectorize text
            X_test = test_df["sentence"]
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
                    y_pred_array = (
                        y_pred.toarray() if hasattr(y_pred, "toarray") else y_pred
                    )

                    # Store essential data in session state
                    st.session_state.eval_y_pred_array = y_pred_array
                    st.session_state.eval_y_test = y_test
                    st.session_state.eval_X_test = X_test

                    # Calculate and store metrics
                    from sklearn.metrics import (
                        f1_score,
                        precision_score,
                        recall_score,
                        hamming_loss,
                        accuracy_score,
                        multilabel_confusion_matrix,
                    )

                    # Store confusion matrices
                    st.session_state.eval_confusion_matrices = (
                        multilabel_confusion_matrix(y_test, y_pred_array)
                    )

                    # Overall metrics
                    st.session_state.eval_accuracy = accuracy_score(
                        y_test, y_pred_array
                    )
                    st.session_state.eval_f1_micro = f1_score(
                        y_test, y_pred_array, average="micro"
                    )
                    st.session_state.eval_f1_macro = f1_score(
                        y_test, y_pred_array, average="macro"
                    )
                    st.session_state.eval_h_loss = hamming_loss(y_test, y_pred_array)

                    # Per-label metrics
                    per_label_metrics = pd.DataFrame(index=label_columns)
                    for i, label in enumerate(label_columns):
                        per_label_metrics.loc[label, "Precision"] = precision_score(
                            y_test[label], y_pred_array[:, i]
                        )
                        per_label_metrics.loc[label, "Recall"] = recall_score(
                            y_test[label], y_pred_array[:, i]
                        )
                        per_label_metrics.loc[label, "F1 Score"] = f1_score(
                            y_test[label], y_pred_array[:, i]
                        )
                        per_label_metrics.loc[label, "Support"] = y_test[label].sum()

                    st.session_state.eval_per_label_metrics = per_label_metrics

                    # Aspect metrics
                    aspects = ["fuel", "machine", "part"]
                    aspect_metrics = pd.DataFrame(index=aspects)

                    for aspect in aspects:
                        aspect_labels = [
                            col for col in label_columns if col.startswith(f"{aspect}_")
                        ]
                        aspect_indices = [
                            label_columns.index(label) for label in aspect_labels
                        ]

                        y_true_aspect = y_test[aspect_labels].values
                        y_pred_aspect = y_pred_array[:, aspect_indices]

                        aspect_metrics.loc[aspect, "Precision"] = precision_score(
                            y_true_aspect, y_pred_aspect, average="micro"
                        )
                        aspect_metrics.loc[aspect, "Recall"] = recall_score(
                            y_true_aspect, y_pred_aspect, average="micro"
                        )
                        aspect_metrics.loc[aspect, "F1 Score"] = f1_score(
                            y_true_aspect, y_pred_aspect, average="micro"
                        )

                    st.session_state.eval_aspect_metrics = aspect_metrics

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
                    st.metric(
                        "F1 Score (micro)", f"{st.session_state.eval_f1_micro:.4f}"
                    )
                    st.progress(st.session_state.eval_f1_micro)

                with col3:
                    st.metric(
                        "F1 Score (macro)", f"{st.session_state.eval_f1_macro:.4f}"
                    )
                    st.progress(st.session_state.eval_f1_macro)

                with col4:
                    # Hamming loss is better when lower, so invert for progress
                    hamming_value = st.session_state.eval_h_loss
                    inverse_progress = max(0, 1 - hamming_value)
                    st.metric("Hamming Loss", f"{hamming_value:.4f}")
                    st.progress(inverse_progress)

                # Detailed metrics sections
                tabs = st.tabs(
                    [
                        "Per-Label Metrics",
                        "Aspect Metrics",
                        "Confusion Matrices",
                        "Sample Predictions",
                    ]
                )

                # Tab 1: Per-label metrics
                with tabs[0]:
                    st.subheader("Performance by Label")

                    # Get the metrics DataFrame
                    metrics_df = st.session_state.eval_per_label_metrics

                    # Add color formatting
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
                        metrics_df.style.format(
                            {
                                "Precision": "{:.4f}",
                                "Recall": "{:.4f}",
                                "F1 Score": "{:.4f}",
                                "Support": "{:.0f}",
                            }
                        ).applymap(
                            color_scale, subset=["Precision", "Recall", "F1 Score"]
                        )
                    )

                    # Visualize F1 scores by label
                    st.subheader("F1 Score by Label")

                    f1_fig, ax = plt.subplots(figsize=(12, 6))
                    metrics_df["F1 Score"].sort_values().plot(kind="barh", ax=ax)
                    plt.xlabel("F1 Score")
                    plt.title("F1 Score Performance by Label")
                    plt.grid(axis="x", linestyle="--", alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(f1_fig)
                    plt.close(f1_fig)

                # Tab 2: Aspect metrics
                with tabs[1]:
                    st.subheader("Performance by Aspect")

                    # Show aspect metrics
                    aspect_df = st.session_state.eval_aspect_metrics
                    st.dataframe(aspect_df.style.format("{:.4f}"))

                    # Visualize metrics by aspect
                    st.subheader("Metrics Comparison by Aspect")

                    # Prepare data for plotting
                    aspect_plot_data = aspect_df.reset_index().melt(
                        id_vars=["index"],
                        value_vars=["Precision", "Recall", "F1 Score"],
                        var_name="Metric",
                        value_name="Score",
                    )

                    # Create grouped bar chart
                    aspect_fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        x="index", y="Score", hue="Metric", data=aspect_plot_data, ax=ax
                    )
                    plt.xlabel("Aspect")
                    plt.ylabel("Score")
                    plt.title("Performance Metrics by Aspect")
                    plt.ylim(0, 1)
                    plt.grid(axis="y", linestyle="--", alpha=0.7)
                    plt.legend(title="Metric")
                    plt.tight_layout()
                    st.pyplot(aspect_fig)
                    plt.close(aspect_fig)

                # Tab 3: Confusion matrices
                with tabs[2]:
                    st.subheader("Confusion Matrices")

                    # Add a selector for confusion matrix display options that uses session state
                    cm_option = st.radio(
                        "Display mode",
                        ["All labels", "By aspect"],
                        index=0 if st.session_state.cm_view == "All labels" else 1,
                        key="cm_display_mode",
                    )

                    # Update the session state
                    st.session_state.cm_view = cm_option

                    # Get confusion matrices
                    mcm = st.session_state.eval_confusion_matrices

                    if cm_option == "All labels":
                        # Create 3 rows of 3 columns for all 9 matrices
                        for row in range(3):
                            cols = st.columns(3)

                            for col in range(3):
                                label_idx = row * 3 + col
                                if label_idx < len(label_columns):
                                    with cols[col]:
                                        fig, ax = plt.subplots(figsize=(4, 3))
                                        sns.heatmap(
                                            mcm[label_idx],
                                            annot=True,
                                            fmt="d",
                                            cmap="Blues",
                                            ax=ax,
                                        )
                                        plt.title(f"{label_columns[label_idx]}")
                                        plt.xlabel("Predicted")
                                        plt.ylabel("Actual")
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                    else:
                        # Show confusion matrices grouped by aspect
                        aspects = ["fuel", "machine", "part"]
                        sentiments = ["negative", "neutral", "positive"]

                        for aspect in aspects:
                            st.write(f"**{aspect.capitalize()} Aspect**")
                            cols = st.columns(3)

                            for i, sentiment in enumerate(sentiments):
                                label = f"{aspect}_{sentiment}"
                                label_idx = label_columns.index(label)

                                with cols[i]:
                                    cm = mcm[label_idx]
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    sns.heatmap(
                                        cm, annot=True, fmt="d", cmap="Blues", ax=ax
                                    )
                                    plt.title(f"{aspect.capitalize()} - {sentiment}")
                                    plt.xlabel("Predicted")
                                    plt.ylabel("Actual")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)

                # Tab 4: Sample predictions
                with tabs[3]:
                    st.subheader("Sample Predictions")

                    # Get test data and predictions
                    X_test = st.session_state.eval_X_test
                    y_test = st.session_state.eval_y_test
                    y_pred_array = st.session_state.eval_y_pred_array

                    # Add a slider to control number of samples shown
                    num_samples = st.slider(
                        "Number of predictions to show",
                        min_value=3,
                        max_value=min(20, len(X_test)),
                        value=10,
                    )

                    # Add a checkbox to filter by correct/incorrect predictions
                    filter_option = st.radio(
                        "Filter predictions",
                        ["Show all", "Show only correct", "Show only incorrect"],
                        index=0,
                    )

                    # Sample indices for predictions
                    all_indices = np.arange(len(X_test))

                    # Determine correct/incorrect predictions
                    if filter_option != "Show all":
                        correct_predictions = []
                        for idx in all_indices:
                            is_correct = np.array_equal(
                                y_test.iloc[idx].values, y_pred_array[idx]
                            )
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

                    # Display samples
                    for i, idx in enumerate(sample_indices):
                        with st.expander(f"Example {i + 1}", expanded=i == 0):
                            text = (
                                X_test.iloc[idx]
                                if hasattr(X_test, "iloc")
                                else X_test[idx]
                            )
                            st.markdown(f"**Text:** {text}")

                            # True and predicted labels
                            true_labels = []
                            pred_labels = []

                            for i, label in enumerate(label_columns):
                                if y_test.iloc[idx, i] == 1:
                                    true_labels.append(label)
                                if y_pred_array[idx, i] == 1:
                                    pred_labels.append(label)

                            # Format for display
                            # Group by aspect for better readability
                            aspects = ["fuel", "machine", "part"]

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**True Labels:**")
                                for aspect in aspects:
                                    aspect_labels = [
                                        label
                                        for label in true_labels
                                        if label.startswith(f"{aspect}_")
                                    ]
                                    if aspect_labels:
                                        label_str = ", ".join(
                                            [l.split("_")[1] for l in aspect_labels]
                                        )
                                        st.markdown(
                                            f"- {aspect.capitalize()}: {label_str}"
                                        )

                            with col2:
                                st.markdown("**Predicted Labels:**")
                                for aspect in aspects:
                                    aspect_labels = [
                                        label
                                        for label in pred_labels
                                        if label.startswith(f"{aspect}_")
                                    ]
                                    if aspect_labels:
                                        label_str = ", ".join(
                                            [l.split("_")[1] for l in aspect_labels]
                                        )
                                        st.markdown(
                                            f"- {aspect.capitalize()}: {label_str}"
                                        )

                            # Check if prediction is correct
                            is_correct = np.array_equal(
                                y_test.iloc[idx].values, y_pred_array[idx]
                            )
                            st.markdown(
                                f"**Prediction Status:** {'✅ Correct' if is_correct else '❌ Incorrect'}"
                            )

                # Download evaluation report
                st.subheader("Export Evaluation Results")

                # Columns for download options
                col1, col2 = st.columns(2)

                with col1:
                    # Create a report DataFrame for detailed results
                    report_df = pd.DataFrame()
                    report_df["Text"] = X_test.reset_index(drop=True)

                    # Add true and predicted labels
                    for i, label in enumerate(label_columns):
                        report_df[f"{label}_true"] = y_test[label].reset_index(
                            drop=True
                        )
                        report_df[f"{label}_pred"] = y_pred_array[:, i]
                        report_df[f"{label}_match"] = (
                            report_df[f"{label}_true"] == report_df[f"{label}_pred"]
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
                        "Metric": [
                            "Accuracy",
                            "F1 (micro)",
                            "F1 (macro)",
                            "Hamming Loss",
                        ],
                        "Score": [
                            st.session_state.eval_accuracy,
                            st.session_state.eval_f1_micro,
                            st.session_state.eval_f1_macro,
                            st.session_state.eval_h_loss,
                        ],
                    }

                    # Add per-label F1 scores to summary
                    for label in label_columns:
                        summary_dict["Metric"].append(f"F1 - {label}")
                        summary_dict["Score"].append(
                            st.session_state.eval_per_label_metrics.loc[
                                label, "F1 Score"
                            ]
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
# Page 3: Prediction
elif page == "Prediction":
    st.header("Make Predictions")

    # Input text
    user_input = st.text_area(
        "tuliskan review (terserah apa saja):", "Avanza bahan bakar nya boros banget"
    )

    # Check if we have trained model stored in session state
    if st.session_state.trained_model is None:
        st.warning(
            "No trained model found. Please train a model in the 'Model Training' page first."
        )
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
                "No trained model found. Training a default model Random Forest...."
            )

            # Define label columns
            label_columns = [
                "fuel_negative",
                "fuel_neutral",
                "fuel_positive",
                "machine_negative",
                "machine_neutral",
                "machine_positive",
                "part_negative",
                "part_neutral",
                "part_positive",
            ]

            # Create multilabel target for training
            y_multilabel = pd.DataFrame()
            for sentiment in ["fuel", "machine", "part"]:
                for label in ["negative", "neutral", "positive"]:
                    col_name = f"{sentiment}_{label}"
                    y_multilabel[col_name] = (df[sentiment] == label).astype(int)

            # Split data
            X_train, _, y_train, _ = train_test_split(
                df["sentence"], y_multilabel, test_size=0.2, random_state=42
            )

            # Vectorize properly
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            input_tfidf = vectorizer.transform([preprocessed_input])

            # Train model
            model = BinaryRelevance(
                classifier=RandomForestClassifier(n_estimators=100, random_state=42)
            )
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
            fuel_preds = [col for col in results if col.startswith("fuel_")]
            machine_preds = [col for col in results if col.startswith("machine_")]
            part_preds = [col for col in results if col.startswith("part_")]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Fuel sentiment:**")
                if fuel_preds:
                    for pred in fuel_preds:
                        st.markdown(
                            f"<span style='color: red;'>- {pred.replace('fuel_', '')}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No prediction")

            with col2:
                st.write("**Machine sentiment:**")
                if machine_preds:
                    for pred in machine_preds:
                        st.markdown(
                            f"<span style='color: green;'>- {pred.replace('machine_', '')}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No prediction")

            with col3:
                st.write("**Part sentiment:**")
                if part_preds:
                    for pred in part_preds:
                        st.markdown(
                            f"<span style='color: blue;'>- {pred.replace('part_', '')}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No prediction")
        else:
            st.write("No labels predicted.")
