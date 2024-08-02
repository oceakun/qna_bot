import os

class Config:
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_TRACK_NOTIFICATIONS = False 
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT', 5432)
    API_URL = os.getenv('API_URL')
    TABLE_NAME = os.getenv('TABLE_NAME')
    
    if all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST]):
        SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///default.db')

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DEV_DATABASE_URL', Config.SQLALCHEMY_DATABASE_URI)

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv('PROD_DATABASE_URL', Config.SQLALCHEMY_DATABASE_URI)

class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('TEST_DATABASE_URL', Config.SQLALCHEMY_DATABASE_URI)
