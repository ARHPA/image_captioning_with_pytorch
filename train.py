import data_sets
from Model import Model
import torch.utils.data
from torch import nn

batch_size = 2
num_epoch = 1
max_caption_length = 40
num_words = 2000

model = Model(batch_size=batch_size, max_caption_length=max_caption_length, num_words=num_words)
# optimizer
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                             lr=0.0001)
# Loss function
criterion = nn.CrossEntropyLoss()
# Custom dataloader
train_loader = data.get_loader(batch_size=batch_size, max_qst_length=max_caption_length)

model.train()
for i in range(num_epoch):
    for batch_idx, batch_sample in enumerate(train_loader):
        image, caption = batch_sample
        scores, caps_sorted = model(image, caption)
        scores = scores.view(batch_size, num_words, max_caption_length)
        loss = criterion(scores, caps_sorted)
        if batch_idx % 10 == 0:
            print("epoch: ", i, ", bach num: ", batch_idx, ", loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()

