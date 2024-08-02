import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
from config import Config
import tfidf_lsa 
import word2vec
import os

def retrieve_similar_content_llm(query):
    def get_most_similar_records(query_embedding):
        conn = psycopg2.connect(
         dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
            )
        
        embedding_array = np.array(query_embedding)

        register_vector(conn)

        cur = conn.cursor()

        table_name="simple_medical_qa_2"
        cur.execute(f"SELECT question, answer, 1 - (embeddings <=> %s::vector) AS confidence FROM {table_name} ORDER BY embeddings <-> %s::vector LIMIT 3", (embedding_array, embedding_array))

        top_docs = cur.fetchall()
        return top_docs

    def get_gte_large_embeddings(text):
            tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
            model = AutoModel.from_pretrained("thenlper/gte-large")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embedding_1d = embeddings.flatten()
            return embedding_1d
    
    return get_most_similar_records(get_gte_large_embeddings(query))


def retrieve_similar_content_tfidf(query):
    def get_most_similar_records(query_embedding):
        conn = psycopg2.connect(
         dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
            )
        
        embedding_array = np.array(query_embedding)

        register_vector(conn)

        cur = conn.cursor()

        table_name="tfdif_lsa_medical_qa"
        cur.execute(f"SELECT question, answer, 1 - (embeddings <=> %s::vector) AS confidence FROM {table_name} ORDER BY embeddings <-> %s::vector LIMIT 3", (embedding_array, embedding_array))

        top_docs = cur.fetchall()
        return top_docs
    
    def get_tfidf_lsa_embeddings(text):
        model_path = 'app/tfidf_lsa_model.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train and save the model first.")
        embedding_creator = tfidf_lsa.EmbeddingCreator.load(model_path)
        return tfidf_lsa.get_embeddings(text, embedding_creator)

    return get_most_similar_records(get_tfidf_lsa_embeddings(query))


def retrieve_similar_content_word2vec(query):
    def get_most_similar_records(query_embedding):
        conn = psycopg2.connect(
         dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
            )
        
        embedding_array = np.array(query_embedding)

        register_vector(conn)

        cur = conn.cursor()

        table_name="word_2_vec_medical_qa"
        cur.execute(f"SELECT question, answer, 1 - (embeddings <=> %s::vector) AS confidence FROM {table_name} ORDER BY embeddings <-> %s::vector LIMIT 3", (embedding_array, embedding_array))

        top_docs = cur.fetchall()
        return top_docs

    def get_word2vec_embeddings(text):
            model_path = "app/faq_word2vec.model"
            return word2vec.get_embeddings(model_path, text)
    
    return get_most_similar_records(get_word2vec_embeddings(query))