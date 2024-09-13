import pandas as pd
import numpy as np
import torch
import preprossing.ImageEmbedding as imb
import time
import preprossing.TextEmbbeding as textIM
from Autoencoder import getModel
from AutoEncoderModel import AutoEncoders ,MLP



def main():

    #test image :

    imagelist = ["product_image_5096.jpg"]
    image_folder = '/Users/mohammad/Downloads/drive-download-20240912T155737Z-001/'
    current_time1 = time.time()

    pipeline = imb.ImageEmbeddingPipeline(imagelist, image_folder)
    results = pipeline.run()

    current_time2 = time.time()
    print(current_time2 -current_time1)

    print(results["product_image_5096.jpg"])

    modelAddress = "models/AEModel.pth"

    model= getModel(modelAddress)

    print(model.getOutputImageEncoder(torch.tensor(results["product_image_5096.jpg"])))

    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoEncoders(512, 128, 512, 768, 128, 768).to(device)
    if device == 'cpu':
        best_model = torch.load(modelAddress, map_location=torch.device('cpu'))
    else:
        best_model = torch.load(modelAddress)
    """

    """
    modelu = textIM.ModelUtils('/Users/rezamosavi/Documents/image-text/Cods/models/')
    modelu.make_dirs()
    modelu.download_model()
    p = textIM.Preprocess('/Users/rezamosavi/Documents/image-text/Cods/models/')
    """
if __name__ == "__main__":
    main()
