import os
import csv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import clip

class ImageEmbeddingPipeline:
    def __init__(self, dataframe_path, image_folder, bad_images_folder, output_csv):
        self.dataframe_path = dataframe_path
        self.image_folder = image_folder
        self.bad_images_folder = bad_images_folder
        self.output_csv = output_csv

        self.model, self.preprocess = clip.load("ViT-B/32")
        self.model.cuda().eval()

        self.df = pd.read_csv(self.dataframe_path)

        self.bad_images_embeddings = self.embed_images(self.get_image_paths(self.bad_images_folder))

    def get_image_paths(self, folder):
        return [
            os.path.join(folder, img) for img in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, img)) and img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def embed_images(self, img_paths):
        images = []
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(self.preprocess(img))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        if not images:
            return torch.empty(0)

        image_input = torch.tensor(np.stack(images)).cuda()
        return self.model.encode_image(image_input).float()

    def process_batch(self, batch_df):
        """Processes a batch of dataframe rows, embedding images and filtering based on similarities."""
        results = []
        for _, row in batch_df.iterrows():
            # Ensure the downloaded_images value is a string
            if isinstance(row['downloaded_images'], str):
                img_paths = [
                    os.path.join(self.image_folder, img)
                    for img in row['downloaded_images'].split(', ')
                    if os.path.exists(os.path.join(self.image_folder, img))
                ]

                if not img_paths:
                    continue

                embeddings = self.embed_images(img_paths)
                for img_path, embedding in zip(img_paths, embeddings):
                    is_similar = False
                    for bad_embedding in self.bad_images_embeddings:
                        similarity = cosine_similarity(
                            embedding.cpu().detach().numpy().reshape(1, -1),
                            bad_embedding.cpu().detach().numpy().reshape(1, -1)
                        )[0][0]

                        if similarity >= 0.90:
                            is_similar = True
                            break  # Skip saving if similarity is >= 90%

                    if not is_similar:
                        # Save link, image filename, and embedding
                        results.append([row['link'], img_path] + embedding.cpu().detach().numpy().tolist())

        self.save_results(results)

    def save_results(self, data):
        """Saves the filtered data to a CSV file."""
        if not data:
            return

        file_exists = os.path.isfile(self.output_csv)
        with open(self.output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # Write the header only if the file doesn't exist
                header = ['product_link', 'image_filename'] + [f'embedding_{i}' for i in range(len(data[0]) - 2)]
                writer.writerow(header)
            writer.writerows(data)

    def run(self):
        """Runs the pipeline, processing the dataframe in batches."""
        batch_size = 10
        total_batches = len(self.df) // batch_size + (1 if len(self.df) % batch_size > 0 else 0)

        for start in tqdm(range(0, len(self.df), batch_size), total=total_batches, desc="Processing Batches"):
            end = min(start + batch_size, len(self.df))
            batch_df = self.df.iloc[start:end]
            self.process_batch(batch_df)

dataframe_path = '/content/drive/MyDrive/data/images-data/pool/man-pool.csv'
image_folder = '/content/drive/MyDrive/data/images-data/pool/product_images-man-pool'
bad_images_folder = '/content/drive/MyDrive/data/images-data/bad-images'
output_csv = '/content/drive/MyDrive/all-pool.csv'

pipeline = ImageEmbeddingPipeline(dataframe_path, image_folder, bad_images_folder, output_csv)
pipeline.run()
