import faiss
import numpy as np
import pandas as pd
import os

from numpy.core.numeric import indices


class VectorDatabase:
    def __init__(self, vec_path='Cods/VectorDatabase/vector_index.faiss', data_path='Cods/VectorDatabase/search.csv'):
        self.vec_path = vec_path
        self.data_path = data_path
        self.index = self.loadVecDatabase()
        self.search = self.loadSearchDatabase()

    def loadVecDatabase(self):
        if os.path.exists(self.vec_path):
            return faiss.read_index(self.vec_path)
        else:
            print('VectorDatabase does not exist')
            return None

    def loadSearchDatabase(self):
        if os.path.exists(self.data_path):
            return pd.read_csv(self.data_path)
        else:
            print('SearchDatabase does not exist')
            return None

    def searchVecDatabase(self, query_embedding):
        k = 1000  # Number of nearest neighbors to retrieve
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)

        return distances, indices

    def findSimilarProducts(self, embedded_vector, limit=10):
        query_embedding = embedded_vector.cpu().detach().numpy()
        _, indices = self.searchVecDatabase(query_embedding)
        finded = []
        for idx in indices[0]:
            find_link = self.search.loc[idx, 'product_link']
            if find_link not in finded:
                finded.append(find_link)
                if len(finded) == limit:
                    break

        return finded