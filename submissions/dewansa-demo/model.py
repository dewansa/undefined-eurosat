"""Baseline: a tiny convolutional network, trained briefly.

It is deliberately small and not tuned. It works, and it leaves a lot of room:
more channels, more layers, better learning rates, more epochs, data augmentation...

To make your own submission, copy this folder to submissions/<your-github-name>/
and change anything you like. The only rule for this file: it must define

    def train(train_set, log) -> nn.Module

`train_set` holds the training images. `train_set.images` is one uint8 tensor of
shape (n, 3, 64, 64) with values 0 to 255, and `train_set.labels` holds the labels,
numbers from 0 to 9. `log` draws your training curves on the leaderboard: call it
once per epoch with any of train_loss, valid_loss, train_accuracy, valid_accuracy.

Return your trained model. The scoring script feeds it float test images of shape
(batch, 3, 64, 64) with values 0 to 1, and checks its answers.
"""

import torch
from torch import nn
from torch.nn import functional as F

EPOCHS = 2
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


def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        predictions = torch.cat([model(x.float() / 255).argmax(1) for x in images.split(1000)])
    return (predictions == labels).float().mean().item()


def train(train_set, log) -> nn.Module:
    images, labels = train_set.images, train_set.labels

    # Keep 10% of the training images aside to see how well the model does on
    # images it did not train on. You never get the real test labels.
    order = torch.randperm(len(images))
    n_valid = len(images) // 10
    valid_images, valid_labels = images[order[:n_valid]], labels[order[:n_valid]]
    train_images, train_labels = images[order[n_valid:]], labels[order[n_valid:]]

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
        valid_accuracy = accuracy(model, valid_images, valid_labels)
        log(train_loss=train_loss, valid_accuracy=valid_accuracy)
        print(f"epoch {epoch + 1}: train loss {train_loss:.3f}, valid acc {valid_accuracy:.1%}")

    return model
