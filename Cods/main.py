from Cods.mainPage import WebApp
from Cods.preprossing.TextEmbbeding import ModelUtils, preprossesor, Preprocess

parsbert_root = "Cods/models/TextEmbedding"  # This path should be added to .gitignore
utils = ModelUtils(parsbert_root)
utils.download_model()

web_app = WebApp()

app = web_app.app