import os
import torch
import numpy as np
from PIL import Image
import clip


class ImageEmbeddingPipeline:
    def __init__(self, input_list, image_folder, batch_size = 10, device=None):
        self.input_list = input_list
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model, self.preprocess = clip.load("ViT-B/32")
        self.model.to(self.device).eval()

    def get_image_paths(self, images_str):
        """Takes a comma-separated string of image filenames and returns their full paths."""
        return [
            os.path.join(self.image_folder, img)
            for img in images_str.split(', ')
            if os.path.exists(os.path.join(self.image_folder, img))
        ]

    def embed_images(self, img_paths):
        """Embeds the given list of image paths."""
        images = []
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(self.preprocess(img))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        if not images:
            return torch.empty(0)

        image_input = torch.tensor(np.stack(images)).to(self.device)
        return self.model.encode_image(image_input).float()

    def process_batch(self, batch):
        """Processes a batch of input, embedding images and returning a list of results."""
        results = []
        for item in batch:
            if isinstance(item, dict) and 'downloaded_images' in item:
                print(item['downloaded_images'])
                img_paths = self.get_image_paths(item['downloaded_images'])

                if not img_paths:
                    continue

                embeddings = self.embed_images(img_paths)
                for img_path, embedding in zip(img_paths, embeddings):
                    # Save link, image filename, and embedding
                    results.append([item['link'], img_path] + embedding.cpu().detach().numpy().tolist())
            else:
                print(f"Invalid item format: {item}, expected a dictionary with 'downloaded_images' key.")
                continue

        return results

    def run(self):
        """Runs the pipeline, processing the input list in batches."""
        batch_size = self.batch_size
        all_results = []

        for start in range(0, len(self.input_list), batch_size):
            end = min(start + batch_size, len(self.input_list))
            batch = self.input_list[start:end]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)

        return all_results
