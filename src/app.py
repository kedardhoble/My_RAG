from flask import Flask, request, jsonify
from retriever import DocumentRetriever
from generator import ResponseGenerator
import yaml

app = Flask(__name__)

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#decine retriever and generator
retriever = DocumentRetriever(config['retriever_model'], config['data_path'])
generator = ResponseGenerator(config['generator_model'])

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data['query']
    documents = retriever.retrieve(query)
    response = generator.generate(query, documents)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
