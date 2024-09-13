from preprossing.TextEmbbeding import ModelUtils, preprossesor, Preprocess

parsbert_root = "models/TextEmbedding"

utils = ModelUtils(parsbert_root)
utils.download_model()

