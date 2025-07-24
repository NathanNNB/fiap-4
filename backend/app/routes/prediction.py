import os
# import joblib
from flask import Blueprint, jsonify, request
# from flask_cors import CORS
# from google.cloud import bigquery
# import numpy as np


prediction = Blueprint("prediction", __name__)
@prediction.route("/", methods=["GET"])
def get_predicition():

    param1 = request.args.get("param1", type=int)
    param2 = request.args.get("param2", type=int)

    return jsonify({"param1": param1, "param2": param2})

