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
        print(distances)
        return distances, indices

    def findSimilarProducts(self, embedded_vector, limit=10):
        query_embedding = embedded_vector.cpu().detach().numpy()
        _, indices = self.searchVecDatabase(query_embedding)
        finded = []
        for idx in indices[0]:
            find_link = self.search.loc[idx, 'product_link']
            if find_link not in finded:
                # base_link = 'https://storage.googleapis.com/kagglesdsdata/datasets/1808165/2949265/images/000000000073.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20241008%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241008T142737Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=1ca54b2bc1d834ee66236fb326b2220f2af1a3307afdaab4b560c74388075a0f80e93d39d1091cde03be04232f23fe0c19997a99b7943a64cf866c88932337a8247ba15bbebef79ebc82378119b53fed25cc6e04531ba4b61324074679e9b6679aa61dc76a5d058425af7978a164845a3f58de1cf63c8b2794946b5082eb2afc67b065a61950b91c55b8dce2239b9c01e7ba2ace3223c9e00bc7e06185f9c6741099a22b30463769add626da4640919be3a1ce7746dafddcba1954f0e7878198e5c06f43eb8f7cc1e8c9bf1cc5a3c808fabb9ec4c73fdcc96ed80893c3ea00da8f1a7887dc5024963c873ff0794b04d0f0a80289d932e8dd3acb90b4cdf2e121'
                finded.append(find_link)
                if len(finded) == limit:
                    break

        return finded