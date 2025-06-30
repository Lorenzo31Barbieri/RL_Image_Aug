import torch
from vgg_classifier import VGGClassifierWrapper

dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, 3 canali, 224x224
model_test = VGGClassifierWrapper()
encoder_output = model_test.encoder(dummy_input)
print(encoder_output.shape) # Dovrebbe stampare torch.Size([1, 25088])