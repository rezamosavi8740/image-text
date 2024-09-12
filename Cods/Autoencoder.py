import AutoEncoderModel as AM
from Meters import AverageMeter
import torch


class getModel():
    def __int__(self ,modelAddress = None ,device = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AM.AutoEncoders(512, 128, 512, 768, 128, 768).to(device)
        self.modelAddress = modelAddress
        self.loadedModel = self.loadModel()

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
        return torch.load(self.modelAddress)

    def getOutput(self ,embeddingVector):
        pass

