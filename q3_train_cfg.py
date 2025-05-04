from torch.utils.data import DataLoader

from cfg_utils.args import args
from cfg_utils.dataset import MNISTDataset
from cfg_utils.unet import UNet_conditional
from q3_cfg_diffusion import CFGDiffusion
from q3_trainer_cfg import Trainer


def train():
    """Trains the classifier-free guidance model."""

    # first get model
    eps_model = UNet_conditional(c_in=1, c_out=1, num_classes=10)

    # now make the CFG diffusion model trainer
    diffusion_model = CFGDiffusion(
        eps_model=eps_model, n_steps=args.n_steps, device=args.device
    )
    trainer = Trainer(args, eps_model, diffusion_model)

    # now get data
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


if __name__ == "__main__":
    train()
