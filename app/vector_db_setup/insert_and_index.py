import pandas as pd
import numpy as np
import psycopg2
import math
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def insert_data_and_create_index_for_complex_data():
    df = pd.read_csv('first_100_records.csv')
    
    df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
    
    conn = psycopg2.connect(
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
    )
    
    try:
        cur = conn.cursor()
        
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        
        register_vector(conn)
        
        table_create_command = """
            CREATE TABLE IF NOT EXISTS medical_qa (
                id BIGSERIAL PRIMARY KEY,
                qtype TEXT,
                question TEXT,
                answer TEXT,
                tokens INTEGER,
                embeddings VECTOR(1024)
            );
        """
        cur.execute(table_create_command)
        conn.commit()
        
        data_list = [
            (row['qtype'], row['question'], row['answer'], int(row['tokens']), row['embeddings'])
            for index, row in df.iterrows()
        ]
        
        print("Trying insertion")
        execute_values(cur, """
            INSERT INTO medical_qa (qtype, question, answer, tokens, embeddings) 
            VALUES %s""", data_list)
        conn.commit()
        
        cur.execute("SELECT COUNT(*) as cnt FROM medical_qa;")
        num_records = cur.fetchone()[0]
        print("Number of vector records in table: ", num_records,"\n")
        
        cur.execute("SELECT * FROM medical_qa LIMIT 1;")
        records = cur.fetchall()
        print("First record in table: ", records)
        
        num_lists = max(10, min(num_records // 1000, int(math.sqrt(num_records))))
        cur.execute(f'CREATE INDEX ON medical_qa USING ivfflat (embeddings vector_cosine_ops) WITH (lists = {num_lists});')
        conn.commit()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def insert_data_and_create_index_for_simple_data():
    df = pd.read_csv('simple_data.csv')
    
    df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
    
    conn = psycopg2.connect(
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST
    )
    
    try:
        cur = conn.cursor()
        
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        
        register_vector(conn)
        
        table_create_command = """
            CREATE TABLE IF NOT EXISTS simple_medical_qa (
                id BIGSERIAL PRIMARY KEY,
                question TEXT,
                answer TEXT,
                tokens INTEGER,
                embeddings VECTOR(1024)
            );
        """
        cur.execute(table_create_command)
        conn.commit()
        
        data_list = [
            (row['question'], row['answer'], int(row['tokens']), row['embeddings'])
            for index, row in df.iterrows()
        ]
        
        print("Trying insertion")
        execute_values(cur, """
            INSERT INTO simple_medical_qa (question, answer, tokens, embeddings) 
            VALUES %s""", data_list)
        conn.commit()
        
        # Count records
        cur.execute("SELECT COUNT(*) as cnt FROM simple_medical_qa;")
        num_records = cur.fetchone()[0]
        print("Number of vector records in table: ", num_records,"\n")
        
        # Fetch first record
        cur.execute("SELECT * FROM simple_medical_qa LIMIT 1;")
        records = cur.fetchall()
        print("First record in table: ", records)
        
        # Index
        num_lists = max(10, min(num_records // 1000, int(math.sqrt(num_records))))
        cur.execute(f'CREATE INDEX ON simple_medical_qa USING ivfflat (embeddings vector_cosine_ops) WITH (lists = {num_lists});')
        conn.commit()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


insert_data_and_create_index_for_simple_data()