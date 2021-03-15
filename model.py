import torch as T
import torch.nn as nn
import torchvision.models as models
from config import Hyper, Constants

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, Hyper.embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(Hyper.dropout_rate)  # Consider other regulisers

    def forward(self, images):
        features = self.resnet(images)
        for name, param in self.resnet.named_parameters():
            param.requires_grad = self.set_grad(name)
        return self.dropout(self.relu(features))

    def set_grad(self, name):
        if "fc.weight" in name or "fc.bias" in name:
            return True

        return Constants.train_CNN

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(Constants.vocab_size, Hyper.embed_size)
        self.lstm = nn.LSTM(Hyper.embed_size, Hyper.hidden_size, Hyper.num_layers)
        self.linear = nn.Linear(Hyper.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(Hyper.dropout_rate)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = T.cat((features.unsqueeze(0), embeddings), dim = 0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, vocabulary):
        super(CNNtoRNN, self).__init__()
        self.vocabulary = vocabulary
        vocab_size = len(vocabulary.itos)
        self.encoderCNN = EncoderCNN()
        self.decoderRNN = DecoderRNN(vocab_size)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image):
        result_caption = []
        with T.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None
            for _ in range(Constants.max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.unsqueeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)
                if self.vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        return [self.vocabulary.itos[idx] for idx in result_caption]


