import os
from flask_cors import CORS
from app import create_app

app = create_app()

CORS(app, resources={r"/*": {"origins": "*"}})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))

    if os.getenv("FLASK_ENV") == "local":
        print("Running in LOCAL environment")
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        print("Running in PRODUCTION environment")
        app.run(host='0.0.0.0', port=port)