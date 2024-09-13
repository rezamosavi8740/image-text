import faiss
import numpy as np
import os

def createVecDatabase(embedded_vectors):
    np.random.seed(42)
    embeddings = embedded_vectors.cpu().detach().numpy()
    index = faiss.index_factory(128, 'Flat', faiss.METRIC_INNER_PRODUCT)
    var = index.ntotal
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, 'VectorDatabase/vector_index.faiss')

if __name__ == '__main__':
    createVecDatabase('...')