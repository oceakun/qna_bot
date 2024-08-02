from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
from config import DevelopmentConfig, ProductionConfig, TestingConfig
from query_processing import retrieve_similar_content_llm, retrieve_similar_content_tfidf, retrieve_similar_content_word2vec 

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

config_class = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}.get(os.getenv('FLASK_ENV', 'development'))

app.config.from_object(config_class)

db = SQLAlchemy(app)

app.app_context().push()

@app.route('/api/answer_query', methods=['POST'])
def answer_query():
    data = request.get_json()
    query = data.get('query', '')
    model = data.get('model', '')
    if not query:
        return jsonify({'response': "No query provided"}), 400
    
    if model=="llm":
        similar_content = retrieve_similar_content_llm(query)
    if model=="tfidf":
        similar_content = retrieve_similar_content_tfidf(query)
    if model=="word2vec":
        similar_content = retrieve_similar_content_word2vec(query)

    response_data = [
        {'question': content[0], 'answer': content[1], 'confidence': content[2]}
        for content in similar_content
    ]  
    return jsonify({'response': "Here's similar content", 'data': response_data})

if __name__ == '__main__':
    app.run(debug=True)