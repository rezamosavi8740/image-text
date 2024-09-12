import pandas as pd
import numpy as np
import torch
import preprossing.ImageEmbedding as imb
from datetime import datetime


def main():

    imagelist = ["product_image_727.jpg"]
    image_folder = '/Users/rezamosavi/Desktop/images-data/tee-shirt/product_images-crawled_data-tee-shirts1000,1200/'
    current_time1 = datetime.now()

    pipeline = imb.ImageEmbeddingPipeline(imagelist, image_folder)
    results = pipeline.run()

    current_time2 = datetime.now()
    print(current_time2 -current_time1)


if __name__ == "__main__":
    main()
