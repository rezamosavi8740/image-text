from preprossing.TextEmbbeding import ModelUtils, preprossesor, Preprocess

parsbert_root = "./Cods/models/TextEmbedding" # This should be changed in the futeure and should be put in .gitignore file

utils = ModelUtils(parsbert_root)
utils.download_model()

