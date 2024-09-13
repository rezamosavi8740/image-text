import os
import torch
import numpy as np
from PIL import Image
import clip


class ImageEmbeddingPipeline:
    def __init__(self, input_list, image_folder, batch_size=10, device=None):
        self.input_list = input_list  # List of image filenames
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model, self.preprocess = clip.load("ViT-B/32")
        self.model.to(self.device).eval()

    def get_image_paths(self):
        """Converts input list of image filenames into full paths."""
        return [
            os.path.join(self.image_folder, img)
            for img in self.input_list
            if os.path.exists(os.path.join(self.image_folder, img))
        ]

    def embed_images(self, img_paths):
        """Embeds the given list of image paths."""
        images = []
        for img_path in img_paths:
            #print(img_path)
            try:
                img = Image.open(self.image_folder+"/"+img_path).convert('RGB')
                #print(img)
                images.append(self.preprocess(img))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        if not images:
            return torch.empty(0)

        image_input = torch.tensor(np.stack(images)).to(self.device)
        return self.model.encode_image(image_input).float()

    def process_batch(self, batch):
        """Processes a batch of image paths and returns a dictionary of embeddings and paths."""
        img_paths = self.get_image_paths()

        if not img_paths:
            return {}

        embeddings = self.embed_images(batch)

        return {img_path: embedding.cpu().detach().numpy().tolist() for img_path, embedding in zip(batch, embeddings)}

    def run(self):
        """Runs the pipeline, processing the input list in batches and returning the results as a dictionary."""
        all_results = {}

        for start in range(0, len(self.input_list), self.batch_size):
            end = min(start + self.batch_size, len(self.input_list))
            batch = self.input_list[start:end]
            batch_results = self.process_batch(batch)
            all_results.update(batch_results)

        return all_results
