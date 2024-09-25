import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify
from embed import embed
from query import query
from get_vector_db import get_vector_db

TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route('/embed', methods=['POST'])
def route_embed():
    if 'file' not in request.files:
        return jsonify({"error": "Partie de fichier manquante"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Pas de fichier sélectionné"}), 400
    
    embedded = embed(file)

    if embedded:
        return jsonify({"message": "Fichier traité"}), 200

    return jsonify({"error": "Erreur avec le fichier. Vérifie le format"}), 400

@app.route('/query', methods=['POST'])
def route_query():
    data = request.get_json()
    response = query(data.get('query'))

    if response:
        return jsonify({"message": response}), 200

    return jsonify({"error": "Un truc s'est mal passé"}), 400

@app.route('/delete', methods=['DELETE'])
def route_delete():
    db = get_vector_db()
    db.delete_collection()

    return jsonify({"message": "La collection a été correctement effacée"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

