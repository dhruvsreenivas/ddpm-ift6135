from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from q1_vae import *

parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
test_loader = DataLoader(
    datasets.MNIST(".", train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=False,
    **kwargs
)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    ## TO DO: Implement the loss function using your functions from q1_solution.py
    ## use the following structure:
    # kl = kl_gaussian_gaussian_analytic(mu_q=?, logvar_q=?, mu_p=?, logvar_p=?).sum()
    # recon_loss = (?).sum()
    # return recon_loss + kl

    # first of all, for KL divergence, we want to compute the KL between N(mu, logvar) and N(0, 1)
    kl = kl_gaussian_gaussian_analytic(
        mu, logvar, torch.zeros_like(mu), torch.zeros_like(logvar)
    ).sum()
    recon_loss = -log_likelihood_bernoulli(recon_x, x).sum()

    return recon_loss + kl


def train(epoch):
    """Trains and evaluates the model. Returns training and evaluation losses for it."""

    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

    # now after the epoch is done, we evaluate the model
    avg_train_loss, avg_eval_loss = evaluate()

    print(
        "====> Epoch: {} Average training loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )
    print(
        "====> Epoch: {} Average validation loss: {:.4f}".format(epoch, avg_eval_loss)
    )

    return avg_train_loss, avg_eval_loss


@torch.no_grad()
def evaluate():
    """Evaluates model on training and test set."""

    model.eval()

    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()

    eval_loss = 0
    for data, _ in test_loader:
        data = data.to(device)

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        eval_loss += loss.item()

    # compute average over length of training dataset.
    return train_loss / len(train_loader.dataset), eval_loss / len(test_loader.dataset)


if __name__ == "__main__":
    training_losses, validation_losses = [], []
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss, eval_loss = train(epoch)

        training_losses.append(train_loss)
        validation_losses.append(eval_loss)

        if epoch == args.epochs:
            print("Final training loss:", train_loss)
            print("Final validation loss:", eval_loss)

    # now we plot the training and validation losses.
    xs = np.arange(args.epochs)

    plt.plot(xs, training_losses, color="red", label="train")
    plt.plot(xs, validation_losses, color="blue", label="eval")

    plt.title("Training/validation losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # save the model
    torch.save(model, "model.pt")

    # save the plot
    plt.tight_layout()
    plt.savefig("images/vae_train_eval.png")
