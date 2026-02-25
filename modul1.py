import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import zipfile
import io
import requests
from os.path import exists
from collections import Counter
import sys
import subprocess
import spacy
# Set page configuration
st.set_page_config(page_title="Word2Vec and Clustering",
                   layout="wide", initial_sidebar_state="expanded")

# Add title and description
st.title("Modul 1 : Feature Engineering and text processing")
st.markdown(
    "Modul 1 dengan Glove, Word2Vec, Bag of Words, TF-IDF dan Clustering")

# Load preprocessed dataset


@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/train_preprocess_stemmed.csv')
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload a CSV file.")
        return pd.DataFrame()


df = load_data()
with st.expander("View Dataset"):
    st.write(df.head())

# Model parameters
st.sidebar.header("Model Parameters")

# Add a dropdown to select the vectorization method
vectorization_method = st.sidebar.radio(
    "Select Vectorization Method",
    ["Word2Vec", "Bag of Words", "TF-IDF", "GloVe"]
)

# Common parameters for all models
vector_size = st.sidebar.slider("Vector Size", min_value=50,
                                max_value=300, value=100, step=50)

# Method-specific parameters
if vectorization_method == "Word2Vec":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        window = st.slider("Window Size", min_value=2,
                           max_value=10, value=5, step=1)
        min_count = st.slider("Min Count", min_value=1,
                              max_value=10, value=1, step=1)
    with col2:
        epochs = st.slider("Epochs", min_value=5,
                           max_value=50, value=10, step=5)
elif vectorization_method in ["Bag of Words", "TF-IDF"]:
    max_features = st.sidebar.slider(
        "Max Features", min_value=100, max_value=10000, value=5000, step=100)
    ngram_range = st.sidebar.selectbox(
        "N-gram Range", options=[(1, 1), (1, 2), (1, 3)], index=0)

# Clustering parameters
st.sidebar.header("Clustering Parameters")
n_clusters = st.sidebar.slider(
    "Number of Clusters", min_value=2, max_value=10, value=3, step=1)
random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)

if st.button(f"Generate {vectorization_method} Vectors and Cluster"):
    with st.spinner(f"Processing {vectorization_method} vectors..."):
        if df.empty:
            st.error("No data available. Please load a dataset first.")
        else:
            sentences = [row.split() for row in df['clean_sentence']]
            doc_vectors = None

            if vectorization_method == "Word2Vec":
                # Word2Vec implementation
                model = Word2Vec(sentences, vector_size=vector_size,
                                 window=window, min_count=min_count, epochs=epochs, seed=random_seed)
                st.success("Word2Vec model trained successfully!")

                # Generate document vectors
                word2id = {word: idx for idx,
                           word in enumerate(model.wv.index_to_key)}
                embedding_matrix = model.wv.vectors
                doc_vectors = np.array(
                    [np.mean([embedding_matrix[word2id[w]] for w in doc.split()
                              if w in word2id], axis=0) for doc in df['clean_sentence']]
                )

            elif vectorization_method == "Bag of Words":
                # BOW implementation
                vectorizer = CountVectorizer(
                    max_features=max_features, ngram_range=ngram_range)
                doc_vectors = vectorizer.fit_transform(
                    df['clean_sentence']).toarray()
                st.success("Bag of Words vectors created successfully!")

                # Get top words for visualization
                feature_names = vectorizer.get_feature_names_out()

                # Display top words with frequencies
                word_freq = np.sum(doc_vectors, axis=0)
                word_freq_df = pd.DataFrame(
                    {'Word': feature_names, 'Frequency': word_freq})
                word_freq_df = word_freq_df.sort_values(
                    'Frequency', ascending=False).head(20)

                st.subheader("Top 20 Words by Frequency")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(word_freq_df['Word'],
                       word_freq_df['Frequency'], color='skyblue')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

                # Add matrix visualization as table
                st.subheader("Bag of Words Matrix Visualization")
                # Select a subset of data for visualization
                num_docs = min(10, doc_vectors.shape[0])
                num_features = min(20, doc_vectors.shape[1])
                matrix_sample = doc_vectors[:num_docs, :num_features]

                # Create DataFrame for the matrix table
                matrix_df = pd.DataFrame(
                    matrix_sample,
                    columns=feature_names[:num_features],
                    index=[f"Doc {i+1}" for i in range(num_docs)]
                )

                # Display the matrix as a table
                st.write("Document-Term Matrix (Sample)")
                st.dataframe(matrix_df.style.highlight_max(
                    axis=1, color='lightgreen'))

            elif vectorization_method == "TF-IDF":
                # TF-IDF implementation
                vectorizer = TfidfVectorizer(
                    max_features=max_features, ngram_range=ngram_range)
                doc_vectors = vectorizer.fit_transform(
                    df['clean_sentence']).toarray()
                st.success("TF-IDF vectors created successfully!")

                # Get feature names for the columns
                feature_names = vectorizer.get_feature_names_out()

                # Add matrix visualization
                st.subheader("TF-IDF Matrix Visualization")
                # Select a subset of data for visualization
                num_docs = min(10, doc_vectors.shape[0])
                num_features = min(20, doc_vectors.shape[1])
                matrix_sample = doc_vectors[:num_docs, :num_features]

                # Create DataFrame for the matrix table
                matrix_df = pd.DataFrame(
                    matrix_sample,
                    columns=feature_names[:num_features],
                    index=[f"Doc {i+1}" for i in range(num_docs)]
                )

                # Display the matrix as a table
                st.write("TF-IDF Document-Term Matrix (Sample)")
                st.dataframe(matrix_df.style.highlight_max(
                    axis=1, color='lightgreen'))

            elif vectorization_method == "GloVe":
                # GloVe implementation
                # Install id_nusantara if not already installed

                with st.status("Setting up Indonesian language model..."):
                    try:
                        # try:
                        nlp = spacy.load("id_nusantara")
                        st.success(
                            "Indonesian language model loaded successfully!")
                        # except OSError:
                        # st.info("Installing Indonesian language model...")
                        # subprocess.check_call(
                        #     [sys.executable, "-m", "pip", "install", "https://huggingface.co/martinastefanoni/id_nusantara/resolve/main/id_nusantara-0.0.0-py3-none-any.whl"])
                        # nlp = spacy.load("id_nusantara")
                        # st.success(
                        #     "Indonesian language model installed successfully!")
                    except Exception as e:
                        st.error(f"Error setting up the model: {e}")
                        st.stop()

                # Create document vectors using spaCy's word vectors
                doc_vectors = []
                for doc in df['clean_sentence']:
                    # Process the document with spaCy
                    processed_doc = nlp(doc)

                    # Get vectors for words in the document
                    if len(processed_doc) > 0 and processed_doc.has_vector:
                        doc_vectors.append(processed_doc.vector)
                    else:
                        # Fallback to zero vector if no vectors available
                        doc_vectors.append(
                            np.zeros(nlp.vocab.vectors.shape[1]))

                doc_vectors = np.array(doc_vectors)
                st.success("Indonesian word vectors created successfully!")

            # Handle NaN values that might occur in document vectors
            doc_vectors = np.nan_to_num(doc_vectors)

            st.write(
                f"{vectorization_method} Document Vectors Shape:", doc_vectors.shape)

            # Visualize document vectors with PCA
            st.subheader(f"Visualize {vectorization_method} Document Vectors")
            pca = PCA(n_components=2)
            reduced_vectors = pca.fit_transform(doc_vectors)

            # Perform KMeans clustering
            km = KMeans(n_clusters=n_clusters, random_state=random_seed)
            cluster_labels = km.fit_predict(doc_vectors)
            df['ClusterLabel'] = cluster_labels

            # Create expanded visualization with words
            plt.figure(figsize=(14, 10))

            # Create scatter plot with cluster colors
            scatter = plt.scatter(
                reduced_vectors[:, 0],
                reduced_vectors[:, 1],
                c=cluster_labels,
                cmap='viridis',
                alpha=0.6,
                s=100,
                edgecolors='w'
            )

            # Add sample words from documents to the plot
            num_samples = min(50, len(df))
            sample_indices = np.random.choice(
                len(df), num_samples, replace=False)

            for idx in sample_indices:
                words = df['clean_sentence'].iloc[idx].split()
                if words:
                    # Select a random word from the document to display
                    display_word = np.random.choice(words)
                    x, y = reduced_vectors[idx]
                    plt.annotate(
                        display_word,
                        (x, y),
                        fontsize=9,
                        alpha=0.8,
                        xytext=(5, 5),
                        textcoords='offset points'
                    )

            # Add cluster centers
            if vectorization_method in ["Word2Vec", "GloVe"]:
                # Project cluster centers to 2D space
                centers_2d = pca.transform(km.cluster_centers_)
                plt.scatter(
                    centers_2d[:, 0],
                    centers_2d[:, 1],
                    s=200,
                    marker='X',
                    c=range(n_clusters),
                    cmap='viridis',
                    edgecolors='k',
                    alpha=0.8
                )

                # Label cluster centers
                for i, (x, y) in enumerate(centers_2d):
                    plt.annotate(
                        f'Cluster {i}',
                        (x, y),
                        fontsize=12,
                        weight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", ec="k", alpha=0.7)
                    )

            plt.colorbar(scatter, label='Cluster')
            plt.title(
                f"{vectorization_method} Document Vectors with Words and Clusters", fontsize=14)
            plt.tight_layout()
            st.pyplot(plt)

            # Display clustered data
            st.subheader("Clustered Data Results")
            cluster_counts = df['ClusterLabel'].value_counts().sort_index()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("Cluster Distribution:")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(
                    cluster_counts,
                    labels=[f"Cluster {i}" for i in range(n_clusters)],
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=True,
                    explode=[0.05] * n_clusters
                )
                ax.axis('equal')
                plt.title("Documents per Cluster", fontsize=14)
                st.pyplot(fig)

            with col2:
                # Show samples from each cluster
                for cluster in range(n_clusters):
                    with st.expander(f"Cluster {cluster} Samples ({cluster_counts[cluster]} documents)"):
                        cluster_samples = df[df['ClusterLabel'] == cluster].head(
                            3)
                        st.write(cluster_samples[['clean_sentence']])

                        # Get most common words in this cluster
                        cluster_text = ' '.join(
                            df[df['ClusterLabel'] == cluster]['clean_sentence'])
                        words = cluster_text.split()
                        word_counts = Counter(words).most_common(10)
                        st.write("Most common words in this cluster:")
                        st.write(
                            ", ".join([f"{w} ({c})" for w, c in word_counts]))
