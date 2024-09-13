import AutoEncoderModel as AM
from Meters import AverageMeter
import torch.nn.functional as F
import torch


class getModel():
    def __init__(self ,modelAddress ,input_shape_image=512 ,latent_shape_image=128 ,input_shape_text=768 ,latent_shape_text=128, modelType="Autoencoder" ,device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.modelType = modelType
        self.modelAddress = modelAddress
        self.input_shape_image = input_shape_image
        self.latent_shape_image = latent_shape_image
        self.input_shape_text = input_shape_text
        self.latent_shape_text = latent_shape_text
        self.defineModel()
        self.loadedModel = self.loadModel()

    def defineModel(self):
        if self.modelType == "imageEncoder":
            self.model = AM.MLP(self.input_shape_image, [1024, 512, 256], self.latent_shape_image, F.relu, [1, 2], weight_init='He').to(self.device)
        elif self.modelType == "textencoder":
            AM.MLP(self.input_shape_text, [1024, 512, 256], self.latent_shape_text, F.relu, [1, 2], weight_init='He').to(self.device)
        else:
            self.model = AM.AutoEncoders(512, 128, 512, 768, 128, 768).to(self.device)

    def evaluateLoss(self , test_loader, loss_fn):
        self.loadedModel.eval()
        loss_eval = AverageMeter()

        with torch.inference_mode():
            for inputs1, inputs2 in test_loader:
                inputs1 = inputs1.to(self.device)
                inputs2 = inputs2.to(self.device)

                # Forward pass
                encoded_image, encoded_text, decoded_image, decoded_text = self.loadedModel(inputs1, inputs2)

                # Compute losses
                latent_loss = loss_fn(encoded_image, encoded_text)
                model1_loss = loss_fn(decoded_image, inputs1)
                model2_loss = loss_fn(decoded_text, inputs2)

                loss = latent_loss + model1_loss + model2_loss

                loss_eval.update(loss.item(), n=len(inputs1))

        return loss_eval.avg

    def loadModel(self):
        if self.device == 'cpu':
            return torch.load(self.modelAddress, map_location=torch.device('cpu'))
        else:
            return torch.load(self.modelAddress, map_location=torch.device('cuda'))

    def getOutputAotuEncoder(self ,embeddingVectorImage ,embeddingVectorText):
        return self.loadedModel(embeddingVectorImage.to(self.device), embeddingVectorText.to(self.device))

    def getOutputImageEncoder(self ,embeddingVector):

        print(self.loadedModel)
        return self.loadedModel(embeddingVector.to(self.device))

    def getOutputTextEncoder(self ,embeddingVector):
        return self.loadedModel(embeddingVector.to(self.device))
