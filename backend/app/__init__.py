from flask import Flask
from flask_cors import CORS
from .routes.prediction import prediction

def create_app():
    app = Flask(__name__)
    
    # CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    # Registro das rotas

    app.register_blueprint(prediction, url_prefix="/prediction")

    return app