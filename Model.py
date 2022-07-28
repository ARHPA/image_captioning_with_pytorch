import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class Encoder(nn.Module):
    def __init__(self, output_size):
        """
        :param output_size: size of model output channels
        """
        super(Encoder, self).__init__()

        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(model.children())[:-2])
        self.adaptive = nn.AdaptiveAvgPool2d((output_size, output_size))

        for param in self.resnet.parameters():
            param.requires_grad = False
        for layer in list(self.resnet.children())[5:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, image):
        """
        :param image: input images  (batch_size, 3, images_input_size, images_input_size)
        :return: encoding output    (batch_size, output_size, output_size, num_output_channels)
        """
        res_out = self.resnet(image)    # [batch_size, num_output_channels, resnet_output_size, resnet_output_size]
        out = self.adaptive(res_out)    # [batch_size, num_output_channels, output_size, output_size]
        out = out.permute(0, 2, 3, 1)   # [batch_size, output_size, output_size, num_output_channels]
        return out


class Attention(nn.Module):
    def __init__(self, encode_dim, hidden_dim, attention_dim):
        """
        :param encode_dim: num chanel of encode output
        :param hidden_dim: dimension of lstm hidden state
        :param attention_dim: dimension of attention
        """
        super(Attention, self).__init__()
        self.encoder_attention = nn.Linear(encode_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encode_out, hidden_out):
        """
        :param encode_out: output of encode model   (batch_size, num_pixels, encode_dim)
        :param hidden_out: output of lstm cell      (batch_size, hidden_dim)
        :return: attention weighted encoding        (batch_size, encode_dim)
        """
        att_encode = self.encoder_attention(encode_out)  # [batch_size, num_pixels, attention_dim]
        att_hidden = self.hidden_attention(hidden_out)  # [batch_size, attention_dim]
        relu_out = self.relu(att_encode + att_hidden.unsqueeze(dim=1))  # [batch_size, num_pixels, attention_dim]
        att = self.attention(relu_out).squeeze(dim=2)  # [batch_size, num_pixels]
        alpha = self.softmax(att)  # [batch_size, num_pixels]
        weighted_encoding = (encode_out * alpha.unsqueeze(dim=2)).sum(dim=1)  # [batch_size, encode_dim]
        return weighted_encoding


class Decoder(nn.Module):
    def __init__(self, batch_size, encode_dim, hidden_dim, embed_dim, attention_dim, num_words, max_caption_length, drop_out=0.5):
        """
        :param encode_dim: num chanel of encode output
        :param hidden_dim: dimension of lstm hidden state
        :param embed_dim: dimension of word embedding
        :param attention_dim: dimension of attention layer
        :param num_words: number of word in embedding
        :param drop_out: value of p in dropout layer
        :param max_caption_length: maximum length of captions
        """

        self.batch_size = batch_size
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.max_caption_length = max_caption_length
        self.num_words = num_words

        super(Decoder, self).__init__()
        self.decode_step = nn.LSTMCell(input_size=encode_dim+embed_dim, hidden_size=hidden_dim)
        self.attention = Attention(encode_dim=encode_dim, hidden_dim=hidden_dim, attention_dim=attention_dim)
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()
        self.create_h = nn.Linear(encode_dim, hidden_dim)
        self.create_c = nn.Linear(encode_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_words)
        self.softmax = nn.Softmax(dim=1)
        self.gate_fc = nn.Linear(hidden_dim, encode_dim)

    def forward(self, encode_out, captions):
        """
        :param encode_out: output of encoder            (batch_size, num_pixels, encode_dim)
        :param captions: image captions                 (batch_size, max_caption_length)
        :return: prediction of one batch_size           (batch_size, max_caption_length, num_words)
        """
        embed_captions = self.embedding(captions)           # [batch_size, max_caption_length, embed_dim]

        mean_encoder_out = encode_out.mean(dim=1)           # [batch_size, encode_dim]
        h = self.create_h(mean_encoder_out)                 # [batch_size, hidden_dim]
        c = self.create_c(mean_encoder_out)                 # [batch_size, hidden_dim]
        for_length = self.max_caption_length - 1

        predicts = torch.zeros((self.batch_size, self.max_caption_length, self.num_words))    # [batch_size, max_caption_length, num_words]
        for t in range(for_length):
            att_weight = self.attention(encode_out, h)                                        # [batch_size, encode_dim]
            gate = self.gate_fc(self.sigmoid(h))                                              # [batch_size, encode_dim]
            decode_input = torch.cat((embed_captions[:, t, :], att_weight * gate), dim=1)     # [batch_size, encode_dim + embed_dim]
            h, c = self.decode_step(decode_input, (h, c))                                     # [batch_size, hidden_dim]
            predict = self.fc(self.dropout(h))                                                # [batch_size, num_words]
            predicts[:, t, :] = predict                                                       # [batch_size, 1, num_words]

        return predicts                                                                       # [batch_size, max_caption_length, num_words]


class Model(nn.Module):
    def __init__(self, encoder_output_size=14, batch_size=256, encode_dim=512, hidden_dim=128, embed_dim=300,
                 attention_dim=256, num_words=2000, max_caption_length=30):

        self.batch_size = batch_size
        self.encode_dim = encode_dim

        super(Model, self).__init__()
        self.encoder = Encoder(output_size=encoder_output_size)
        self.decoder = Decoder(batch_size=batch_size, encode_dim=encode_dim, hidden_dim=hidden_dim, embed_dim=embed_dim,
                               attention_dim=attention_dim, num_words=num_words, max_caption_length=max_caption_length)

    def forward(self, images, captions):
        """
        :param images: input images      (batch_size, 3, images_input_size, images_input_size)
        :param captions: input captions  (batch_size, max_caption_length)
        :return:
        """
        encode_out = self.encoder(images)  # [batch_size, encoder_output_size, encode_dim]
        resize_encode_out = encode_out.view(self.batch_size, -1, self.encode_dim)  # [batch_size, encoder_output_size**2, encode_dim]
        decode_out = self.decoder(resize_encode_out, captions)  # [batch_size, max_caption_length, num_words]
        return decode_out, captions

