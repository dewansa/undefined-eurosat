"""Baseline: a tiny convolutional network, trained briefly.

It is deliberately small and not tuned. It works, and it leaves a lot of room:
more channels, more layers, better learning rates, more epochs, data augmentation...

To make your own submission, copy this folder to submissions/<your-github-name>/
and change anything you like. The only rule for this file: it must define

    def train(train_set, log) -> nn.Module

`train_set` holds the training images. `train_set.images` is one uint8 tensor of
shape (n, 3, 64, 64) with values 0 to 255, and `train_set.labels` holds the labels,
numbers from 0 to 9. `log` draws your training curves on the leaderboard. Give it
validation images once with log.validate_on(images, labels), then call
log(model, train_loss=...) once per epoch: the scorer checks each logged model on
those images after training, off the clock, so watching your model costs no time.

Return your trained model. The scoring script feeds it float test images of shape
(batch, 3, 64, 64) with values 0 to 1, and checks its answers.
"""

import torch
from torch import nn
from torch.nn import functional as F

EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.003


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 8 filters look at 3x3 pixel patches
        nn.ReLU(),
        nn.MaxPool2d(2),  # 64x64 -> 32x32
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 32x32 -> 16x16
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 16x16 -> 8x8
        nn.Flatten(),  # 32 maps of 8x8 -> 2048 numbers
        nn.Linear(32 * 8 * 8, 10),
    )


def train(train_set, log) -> nn.Module:
    images, labels = train_set.images, train_set.labels

    # Keep 10% of the training images aside to see how well the model does on
    # images it did not train on. You never get the real test labels.
    order = torch.randperm(len(images))
    n_valid = len(images) // 10
    valid_images, valid_labels = images[order[:n_valid]], labels[order[:n_valid]]
    train_images, train_labels = images[order[n_valid:]], labels[order[n_valid:]]
    log.validate_on(valid_images, valid_labels)  # checked after training, for free

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        shuffled = torch.randperm(len(train_images))
        for batch in shuffled.split(BATCH_SIZE):
            x = train_images[batch].float() / 255  # convert only this batch to floats
            loss = F.cross_entropy(model(x), train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        train_loss = total_loss / len(train_images)
        log(model, train_loss=train_loss)  # saves a copy; valid_* are measured later
        print(f"epoch {epoch + 1}: train loss {train_loss:.3f}")

    return model
