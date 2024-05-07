import backend
import constants
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 

@app.route('/consultar', methods=['POST'])
def consultar():
    data = request.get_json()
    session_id = data.get('session_id', None)
    query = data.get('query', None)
    response = backend.process_query(session_id, query)
    return response


@app.route('/', methods=['OPTIONS'])
def handle_options():
    return jsonify({'message': 'Success'}), 200

if __name__ == '__main__':
    app.run(debug=True)
