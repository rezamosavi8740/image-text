from preprossing.TextEmbbeding import ModelUtils, preprossesor, Preprocess

import os

def run():
    parsbert_root = "/Users/rezamosavi/Documents/image-text/Cods/models/TextEmbedding"  # This path should be added to .gitignore
    utils = ModelUtils(parsbert_root)
    utils.download_model()
    p = Preprocess(parsbert_root)
    r = p.vectorize("نخ عادی پیراهن")
    print(r)

if __name__ == "__main__":
    if os.path.basename(__file__) == 'setup.py':
        run()


