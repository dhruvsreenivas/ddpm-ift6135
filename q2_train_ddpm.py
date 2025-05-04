import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ddpm_utils.args import args
from ddpm_utils.dataset import MNISTDataset
from ddpm_utils.unet import UNet, load_weights
from q2_ddpm import DenoiseDiffusion
from q2_trainer_ddpm import Trainer


# plot and save the generated images
def plot_and_save_intermediate_samples(images, steps_to_show, n_samples):
    """
    Plot the intermediate steps of the diffusion process
    Args:
        images: List of image tensors at different steps
        steps_to_show: List of steps that were captured
        n_samples: Number of images to show
    """
    # Create a figure with n_samples rows and len(steps_to_show) columns
    plt.figure(figsize=(25, 15 * n_samples))
    fig, axs = plt.subplots(n_samples, len(steps_to_show))
    # Plot each image
    for sample_idx in range(n_samples):
        for step_idx, img in enumerate(images):
            axs[sample_idx, step_idx].imshow(img[sample_idx, 0], cmap="gray")
            step = (
                steps_to_show[step_idx]
                if step_idx < len(steps_to_show)
                else args.n_steps
            )
            axs[sample_idx, step_idx].set_title(
                f" Image {sample_idx} \nt={args.n_steps - step-1}", size=8
            )
            axs[sample_idx, step_idx].axis("off")

    plt.tight_layout()
    plt.savefig("images/ddpm_post_training_intermediate_samples.png")
    plt.close(fig)


def train():
    """Trains the diffusion model."""

    # first set up the actual epsilon model
    eps_model = UNet(c_in=1, c_out=1)
    eps_model = load_weights(eps_model, args.MODEL_PATH)

    # now set up the diffusion model trainer
    diffusion_model = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=args.n_steps,
        device=args.device,
    )
    trainer = Trainer(args, eps_model, diffusion_model)

    # set up data
    dataloader = DataLoader(
        MNISTDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    # train!
    trainer.train(dataloader)

    # once done training, we generate intermediate samples
    steps_to_show = list(range(0, args.n_steps, 100)) + [args.n_steps - 1]
    steps_to_show = [0, 100, 500, 800, 900, 950, 980, 999]
    images = trainer.generate_intermediate_samples(
        n_samples=4, steps_to_show=steps_to_show
    )

    # plot the intermediate samples.
    plot_and_save_intermediate_samples(images, steps_to_show, n_samples=4)


if __name__ == "__main__":
    train()
