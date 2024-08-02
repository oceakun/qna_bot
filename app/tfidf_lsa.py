from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import joblib
import os

class EmbeddingCreator:
    def __init__(self, n_components=190):
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.lsa = TruncatedSVD(n_components=self.n_components)
        self.is_fitted = False

    def fit(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        self.lsa.fit(tfidf_matrix)
        self.is_fitted = True

    def transform(self, text):
        if not self.is_fitted:
            raise ValueError("EmbeddingCreator must be fitted before transform")
        tfidf_vector = self.tfidf.transform([text])
        lsa_vector = self.lsa.transform(tfidf_vector)
        return lsa_vector.flatten()

    def save(self, path):
        joblib.dump((self.tfidf, self.lsa, self.is_fitted), path)

    @classmethod
    def load(cls, path):
        creator = cls()
        creator.tfidf, creator.lsa, creator.is_fitted = joblib.load(path)
        return creator

def get_embeddings(text, embedding_creator=None):
    if embedding_creator is None:
        model_path = 'tfidf_lsa_model.joblib'
        if os.path.exists(model_path):
            embedding_creator = EmbeddingCreator.load(model_path)
        else:
            raise ValueError("No pre-trained TF-IDF LSA model found. Please train the model first.")

    embedding = embedding_creator.transform(text)
    return embedding

# Example usage and model training
if __name__ == "__main__":
    # This part should be run once to train and save the model
    import pandas as pd

    # Load your dataset
    df = pd.read_csv('./vector_db_setup/simple_data_tfidf_lsa.csv')
    texts = df['question']

    # Create and fit the EmbeddingCreator
    creator = EmbeddingCreator(n_components=190)
    creator.fit(texts)

    # Save the fitted model
    creator.save('tfidf_lsa_model.joblib')

    # Example of using the get_embeddings function
    query = "What causes malaria?"
    embedding = get_embeddings(query, creator)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding[:5]}...")  # Print first 5 values