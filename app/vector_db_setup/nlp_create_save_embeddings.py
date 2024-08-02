import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

def process_simple_data():
    print("begins")
    
    # Load the data
    df = pd.read_csv('simple_data.csv', delimiter=',', quotechar='"')
    print(df.head())
    
    df['text'] = df['question']
    
    # Create TF-IDF and LSA pipeline
    tfidf = TfidfVectorizer(stop_words='english')
    lsa = TruncatedSVD(n_components=190)  # Adjust the number of components as needed
    pipeline = make_pipeline(tfidf, lsa)
    
    # Fit and transform the text data
    embeddings = pipeline.fit_transform(df['text'])
    
    # Convert embeddings to list for storage in CSV
    df['embeddings'] = embeddings.tolist()
    
    # Calculate token length (approximation using word count)
    df['tokens'] = df['text'].apply(lambda x: len(x.split()))
    
    # Select and reorder columns
    df_new = df[['question', 'answer', 'tokens', 'embeddings']]
    
    # Save to CSV
    df_new.to_csv('simple_data_tfidf_lsa.csv', index=False)
    print("Process completed. New CSV file created: simple_data_tfidf_lsa.csv")

# Call the function
process_simple_data()