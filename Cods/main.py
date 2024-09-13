import pandas as pd
import numpy as np
import torch
import preprossing.ImageEmbedding as imb
from datetime import datetime
import preprossing.TextEmbbeding as textIM
from Autoencoder import getModel
from AutoEncoderModel import AutoEncoders ,MLP



def main():

    #test image :

    imagelist = ["product_image_727.jpg"]
    image_folder = '/Users/rezamosavi/Desktop/images-data/tee-shirt/product_images-crawled_data-tee-shirts1000,1200/'
    current_time1 = datetime.now()

    pipeline = imb.ImageEmbeddingPipeline(imagelist, image_folder)
    results = pipeline.run()

    current_time2 = datetime.now()
    print(current_time2 -current_time1)

    print(results["product_image_727.jpg"])

    modelAddress = "/Users/rezamosavi/Documents/image-text/Cods/models/AEModel.pt"

    model= getModel(modelAddress ,modelType="imageEncoder")

    print(model.getOutputImageEncoder(torch.tensor(results["product_image_727.jpg"])))

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
