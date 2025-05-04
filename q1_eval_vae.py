import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from q1_train_vae import VAE

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load("model.pt", map_location=device, weights_only=False)
assert isinstance(model, VAE)
model.eval()
latent_dim = 20


@torch.no_grad()
def generate_samples(num_samples=9):
    """Generates samples from the VAE."""

    assert num_samples >= 9

    z = torch.randn(num_samples, latent_dim, dtype=torch.float32, device=device)
    samples = model.decode(z).view(-1, 28, 28)
    samples = samples[:9]

    # now plot the samples
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = samples[i].squeeze().cpu().numpy()

        ax.imshow(img, cmap="gray")  # Display the image in grayscale
        ax.axis("off")  # Hide the axis

    plt.tight_layout()
    plt.savefig("images/vae_samples.png")
    plt.close(fig)


@torch.no_grad()
def visualize_disentanglement(scale, samples_per_latent_factor=5):
    """Checks disentanglement in the latent space and decodes the output."""

    # set up the images list
    images = []

    # start with initial z
    z = torch.randn(
        samples_per_latent_factor, latent_dim, dtype=torch.float32, device=device
    )

    for dim in range(latent_dim):
        # now add `scale` to z at the dim provided
        amt = F.one_hot(torch.tensor(dim).to(device), num_classes=latent_dim) * scale
        amt = amt.unsqueeze(0).float()
        z_modified = z + amt

        # decode the output
        samples = model.decode(z_modified).view(-1, 28, 28)

        # add to the images dir.
        images.append(samples)

    # now we plot
    fig, axes = plt.subplots(latent_dim, samples_per_latent_factor, figsize=(5, 20))
    for i in range(latent_dim):
        for j in range(samples_per_latent_factor):
            img = images[i][j].squeeze().cpu().numpy()

            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")  # Hide the axis

    plt.tight_layout()
    plt.savefig("images/vae_disentangled_samples.png")
    plt.close(fig)


@torch.no_grad()
def visualize_disentanglement_vary_epsilon(init_scale, final_scale):
    """Checks disentanglement in the latent space and decodes the output, where instead of having 5 Gaussian samples, we have 5 different epsilon gradients."""

    # set up the images list
    images = []

    # set up the scaling levels
    scales = np.linspace(init_scale, final_scale, num=5)

    # start with an initial z vector
    z = torch.randn(latent_dim, dtype=torch.float32, device=device)

    # for each dim, we log each of the images appropriately
    for dim in range(latent_dim):
        samples = []

        amt = F.one_hot(torch.tensor(dim).to(device), num_classes=latent_dim)
        amt = amt.unsqueeze(0).float()

        # now, for each scale factor we care about, we interpolate accordingly
        for scale in scales:
            scaled_amt = amt * scale
            z_modified = z + scaled_amt

            # decode the output
            sample = model.decode(z_modified).view(28, 28)
            samples.append(sample)

        # add samples to global images dir
        images.append(samples)

    # now we plot
    fig, axes = plt.subplots(latent_dim, 5, figsize=(5, 20))
    for i in range(latent_dim):
        for j in range(5):
            img = images[i][j].squeeze().cpu().numpy()

            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")  # Hide the axis

    plt.tight_layout()
    plt.savefig("images/vae_disentangled_samples_vary_epsilon.png")
    plt.close(fig)


@torch.no_grad()
def interpolate_latent():
    """Interpolates in the latent space."""

    z0 = torch.randn(latent_dim, dtype=torch.float32, device=device)
    z1 = torch.randn(latent_dim, dtype=torch.float32, device=device)

    samples = []

    for alpha in range(11):
        scaled_alpha = 0.1 * alpha

        # interpolate and decode
        z_alpha = scaled_alpha * z0 + (1.0 - scaled_alpha) * z1
        sample = model.decode(z_alpha).view(28, 28)

        # add to samples list
        samples.append(sample)

    # now plot the samples
    fig, axes = plt.subplots(ncols=11, figsize=(100, 10))
    for i, sample in enumerate(samples):
        sample = sample.cpu().numpy()

        axes[i].imshow(sample, cmap="gray")
        axes[i].axis("off")  # Hide the axis

    plt.tight_layout()
    plt.savefig("images/vae_interpolated_latent_samples.png")
    plt.close(fig)


@torch.no_grad()
def interpolate_data():
    """Interpolates in the data space."""

    z0 = torch.randn(latent_dim, dtype=torch.float32, device=device)
    z1 = torch.randn(latent_dim, dtype=torch.float32, device=device)

    # decode now
    x0 = model.decode(z0).view(28, 28)
    x1 = model.decode(z1).view(28, 28)

    samples = []

    for alpha in range(11):
        scaled_alpha = 0.1 * alpha

        # interpolate now and add to samples list
        x_alpha = scaled_alpha * x0 + (1.0 - scaled_alpha) * x1
        samples.append(x_alpha)

    # now plot the samples
    fig, axes = plt.subplots(ncols=11, figsize=(100, 10))
    for i, sample in enumerate(samples):
        sample = sample.cpu().numpy()

        axes[i].imshow(sample, cmap="gray")
        axes[i].axis("off")  # Hide the axis

    plt.tight_layout()
    plt.savefig("images/vae_interpolated_data_samples.png")
    plt.close(fig)


if __name__ == "__main__":
    # generate_samples()
    # visualize_disentanglement(scale=2.0)
    visualize_disentanglement_vary_epsilon(0.5, 3.0)
    # interpolate_latent()
    # interpolate_data()
