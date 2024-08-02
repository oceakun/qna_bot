import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)

def load_or_train_word2vec(df, vector_size=100, window=5, min_count=1, workers=4):
    """Load pre-trained Word2Vec model or train a new one"""
    try:
        model = Word2Vec.load("faq_word2vec.model")
        print("Loaded pre-trained Word2Vec model.")
    except FileNotFoundError:
        print("Training new Word2Vec model...")
        corpus = [word_tokenize(question.lower()) for question in df['question']]
        model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        model.save("faq_word2vec.model")
        print("New Word2Vec model trained and saved.")
    return model

def get_embeddings(model, question):
    """
    Generate embeddings for a single question
    
    :param model: Either a loaded Word2Vec model or a string path to a saved model
    :param question: The question text to embed
    :return: numpy array of embeddings
    """
    # If model is a string, assume it's a file path and load the model
    if isinstance(model, str):
        try:
            model = Word2Vec.load(model)
        except Exception as e:
            raise ValueError(f"Failed to load model from path: {model}. Error: {str(e)}")
    
    # Ensure model is now a Word2Vec model
    if not isinstance(model, Word2Vec):
        raise TypeError("model must be either a Word2Vec model or a string path to a saved model")
    
    words = word_tokenize(question.lower())
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)    
    if 'question' not in df.columns:
        raise ValueError("CSV file must contain a 'question' column")    
    model = load_or_train_word2vec(df)    
    df['tokens'] = df['question'].apply(lambda q: len(word_tokenize(q)))    
    df['embeddings'] = df['question'].apply(lambda q: get_embeddings(model, q).tolist())
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    input_file = "simple_data.csv"  # Replace with your input file name
    output_file = "simple_data_word2vec_model.csv"  # Replace with your desired output file name
    
    process_csv(input_file, output_file)



# import pandas as pd
# from gensim.models import Word2Vec
# import nltk
# from nltk.tokenize import word_tokenize
# import numpy as np

# nltk.download('punkt')

# def preprocess_data(file_path):
#     df = pd.read_csv(file_path)
#     corpus = []
#     for _, row in df.iterrows():
#         question = word_tokenize(row['question'].lower())
#         corpus.append(question)
#     return corpus, df

# def train_word2vec(corpus, vector_size, window, min_count, workers):
#     model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
#     return model

# def optimize_vector_size(corpus, size_range, window, min_count, workers):
#     best_size = 0
#     best_score = float('-inf')    
#     for size in size_range:
#         model = train_word2vec(corpus, size, window, min_count, workers)
#         score = model.wv.evaluate_word_pairs('wordsim353.tsv')['spearmanr'][0]
#         if score > best_score:
#             best_score = score
#             best_size = size
#     return best_size

# def get_doc_embedding(model, doc):
#     words = word_tokenize(doc.lower())
#     word_vectors = [model.wv[word] for word in words if word in model.wv]
#     return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# if __name__ == "__main__":
#     file_path = 'simple_data.csv'
#     corpus, df = preprocess_data(file_path)
    
#     size_range = range(100, 301, 50)
#     best_size = optimize_vector_size(corpus, size_range, window=5, min_count=1, workers=4)
#     print(f"Optimal vector size: {best_size}")
    
#     final_model = train_word2vec(corpus, vector_size=best_size, window=5, min_count=1, workers=4)
    
#     df['embedding'] = df['question'].apply(lambda x: get_doc_embedding(final_model, x))
    
#     final_model.save("faq_word2vec.model")
#     df.to_pickle("faq_embeddings.pkl")

# print("Model and embeddings saved.")
