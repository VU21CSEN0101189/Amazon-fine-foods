from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained Sentence-BERT model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load embeddings and product data
try:
    with open('product_embeddings.pkl', 'rb') as f:
        df = pickle.load(f)  # Assuming this is a DataFrame
except FileNotFoundError:
    print("Error: 'product_embeddings.pkl' not found. Please ensure the file is in the correct directory.")
    exit(1)

df['imgs'] = df['imgs'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Function to get recommendations
def recommend_products(query, top_k=10):
    if not query.strip():
        return []
    query = query.lower()
    query_embedding = model.encode(query)
    df['similarity'] = df['embeddings'].apply(
        lambda x: cosine_similarity([query_embedding], [x]).flatten()[0]
    )
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_k)
    return recommendations[['title', 'brand', 'category', 'similarity', 'imgs']]

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error_message = None
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query.strip():
            recommendations = recommend_products(query).to_dict(orient='records')
        else:
            error_message = "Query cannot be empty. Please enter a valid product name."
    return render_template('index.html', recommendations=recommendations, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
