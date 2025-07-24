import os
from flask import Blueprint, jsonify, request
from flask_cors import CORS


prediction = Blueprint("prediction", __name__)
CORS(prediction)

@prediction.route("/", methods=["GET"])
def get_predicition():

    param1 = request.args.get("param1", type=int)
    param2 = request.args.get("param2", type=int)

    return jsonify({"param1": param1, "param2": param2})

