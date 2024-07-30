import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
from config import Config

def retrieve_similar_content(query):
    def get_top_similar_docs_complex_data(query_embedding):
        conn = psycopg2.connect(
         dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
            )
        embedding_array = np.array(query_embedding)
        register_vector(conn)
        cur = conn.cursor()
        cur.execute("SELECT qtype, question, answer FROM medical_qa ORDER BY embeddings <-> %s::vector LIMIT 10", (embedding_array,))
        top3_docs = cur.fetchall()
        return top3_docs
    
    def get_top_similar_docs_simple_data(query_embedding):
        conn = psycopg2.connect(
         dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
            )
        embedding_array = np.array(query_embedding)
        register_vector(conn)
        cur = conn.cursor()
        cur.execute("SELECT question, answer FROM simple_medical_qa ORDER BY embeddings <-> %s::vector LIMIT 10", (embedding_array,))
        top3_docs = cur.fetchall()
        return top3_docs

    def get_gte_large_embeddings(text):
            tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
            model = AutoModel.from_pretrained("thenlper/gte-large")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embedding_1d = embeddings.flatten()
            return embedding_1d
    
    return get_top_similar_docs_simple_data(get_gte_large_embeddings(query))
