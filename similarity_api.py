from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Charger le modèle SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/similarity', methods=['POST'])
def calculate_similarity():
    data = request.json
    query = data['query']
    texts = data['texts']

    # Encodez la requête et les textes
    query_embedding = model.encode(query, convert_to_tensor=True)
    text_embeddings = model.encode(texts, convert_to_tensor=True)

    # Calculez les similarités cosinus
    similarities = util.cos_sim(query_embedding, text_embeddings)

    # Retourner les similarités sous forme de liste
    return jsonify({'similarities': similarities.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
