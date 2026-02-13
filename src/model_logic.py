from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def get_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts)

def run_clustering(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels