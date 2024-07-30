import pandas as pd
import numpy as np
import tiktoken
import psycopg2
import math
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def process_csv_and_create_embeddings_table():
    print("begins")
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
    model = AutoModel.from_pretrained("thenlper/gte-large")
    print("model initialized")

    def get_gte_large_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    df = pd.read_csv('first_100_records.csv', delimiter=',', quotechar='"')
    print(df.head())

    def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
        if not string:
            return 0
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    new_list = []
    for i in range(len(df.index)):
        text = df['qtype'][i]+df['question'][i]+df['answer'][i]
        token_len = num_tokens_from_string(text)
        if token_len <= 512:
            new_list.append([df['qtype'][i], df['question'][i], df['answer'][i], token_len, text])
        else:
            start = 0
            ideal_token_size = 512
            ideal_size = int(ideal_token_size // (4/3))
            end = ideal_size
            words = text.split()
            words = [x for x in words if x != ' ']
            total_words = len(words)            
            chunks = total_words // ideal_size
            if total_words % ideal_size != 0:
                chunks += 1
            new_content = []
            for j in range(chunks):
                if end > total_words:
                    end = total_words
                new_content = words[start:end]
                new_content_string = ' '.join(new_content)
                new_content_token_len = num_tokens_from_string(new_content_string)
                if new_content_token_len > 0:
                    new_list.append([df['qtype'][i], df['question'][i], df['answer'][i], new_content_token_len, new_content_string])
                start += ideal_size
                end += ideal_size

    for i in range(len(new_list)):
        text = new_list[i][4]
        embedding = get_gte_large_embeddings(text)
        print("embeddings created for ", i)
        new_list[i][4]= embedding.flatten().tolist()

    df_new = pd.DataFrame(new_list, columns=['qtype', 'question', 'answer', 'tokens', 'embeddings'])
    df_new.head()

    df_new.to_csv('first_100_records.csv', index=False)

    conn = psycopg2.connect(
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)
    conn.commit()

    register_vector(conn)

    table_create_command = """
        CREATE TABLE medical_qa (
            id BIGSERIAL PRIMARY KEY,
            qtype TEXT,
            question TEXT,
            answer TEXT,
            tokens INTEGER,
            embeddings VECTOR(1024)
            );
                """

    cur.execute(table_create_command)
    cur.close()
    conn.commit()

    register_vector(conn)
    cur = conn.cursor()
    data_list = [(row['qtype'], row['question'], row['answer'], int(row['tokens']),  np.array(row['embeddings'])) for index, row in df_new.iterrows()]

    try:
        print("trying insertion")
        execute_values(cur, """
            INSERT INTO medical_qa (qtype, question, answer, tokens, embeddings) 
            VALUES %s""", data_list)
        conn.commit()
    except Exception as e:
        print(f"An error occurred during insertion: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    cur.execute("SELEC COUNT(*) as cnt FROM medical_qa;")
    num_records = cur.fetchone()[0]
    print("Number of vector records in table: ", num_records,"\n")

    cur.execute("SELECT * FROM medical_qa LIMIT 1;")
    records = cur.fetchall()
    print("First record in table: ", records)
    
    num_lists = num_records / 1000
    if num_lists < 10:
        num_lists = 10
    if num_records > 1000000:
        num_lists = math.sqrt(num_records)

    cur.execute(f'CREATE INDEX ON medical_qa USING ivfflat (embeddings vector_cosine_ops) WITH (lists = {num_lists});')
    conn.commit()
    cur.close()
    conn.close()


def process_simple_data():
    print("begins")
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
    model = AutoModel.from_pretrained("thenlper/gte-large")
    print("model initialized")

    def get_gte_large_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    df = pd.read_csv('simple_data.csv', delimiter=',', quotechar='"')
    print(df.head())

    def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
        if not string:
            return 0
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    new_list = []
    for i in range(len(df.index)):
        text = df['question'][i]+df['answer'][i]
        token_len = num_tokens_from_string(text)
        if token_len <= 512:
            new_list.append([df['question'][i], df['answer'][i], token_len, text])
        else:
            start = 0
            ideal_token_size = 512
            ideal_size = int(ideal_token_size // (4/3))
            end = ideal_size
            words = text.split()
            words = [x for x in words if x != ' ']
            total_words = len(words)            
            chunks = total_words // ideal_size
            if total_words % ideal_size != 0:
                chunks += 1
            new_content = []
            for j in range(chunks):
                if end > total_words:
                    end = total_words
                new_content = words[start:end]
                new_content_string = ' '.join(new_content)
                new_content_token_len = num_tokens_from_string(new_content_string)
                if new_content_token_len > 0:
                    new_list.append([df['question'][i], df['answer'][i], new_content_token_len, new_content_string])
                start += ideal_size
                end += ideal_size

    for i in range(len(new_list)):
        text = new_list[i][3]
        embedding = get_gte_large_embeddings(text)
        print("embeddings created for ", i)
        new_list[i][3]= embedding.flatten().tolist()

    df_new = pd.DataFrame(new_list, columns=['question', 'answer', 'tokens', 'embeddings'])
    df_new.head()

    df_new.to_csv('simple_data.csv', index=False)

    conn = psycopg2.connect(
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)
    conn.commit()

    register_vector(conn)

    table_create_command = """
        CREATE TABLE simple_medical_qa (
            id BIGSERIAL PRIMARY KEY,
            question TEXT,
            answer TEXT,
            tokens INTEGER,
            embeddings VECTOR(1024)
            );
                """

    cur.execute(table_create_command)
    cur.close()
    conn.commit()

    register_vector(conn)
    cur = conn.cursor()
    data_list = [( row['question'], row['answer'], int(row['tokens']),  np.array(row['embeddings'])) for index, row in df_new.iterrows()]

    try:
        print("trying insertion")
        execute_values(cur, """
            INSERT INTO simple_medical_qa (question, answer, tokens, embeddings) 
            VALUES %s""", data_list)
        conn.commit()
    except Exception as e:
        print(f"An error occurred during insertion: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    cur.execute("SELECT COUNT(*) as cnt FROM simple_medical_qa;")
    num_records = cur.fetchone()[0]
    print("Number of vector records in table: ", num_records,"\n")

    cur.execute("SELECT * FROM simple_medical_qa LIMIT 1;")
    records = cur.fetchall()
    print("First record in table: ", records)
    
    num_lists = num_records / 1000
    if num_lists < 10:
        num_lists = 10
    if num_records > 1000000:
        num_lists = math.sqrt(num_records)

    cur.execute(f'CREATE INDEX ON simple_medical_qa USING ivfflat (embeddings vector_cosine_ops) WITH (lists = {num_lists});')
    conn.commit()
    cur.close()
    conn.close()

process_simple_data()