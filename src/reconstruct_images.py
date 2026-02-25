import matplotlib.pyplot as plt
import numpy as np
from array import array
from os.path  import join
import torch
import random

from .data_loader import MnistDataloader
from .model import VariationalAutoEncoder
from .config import Config

config = Config()
training_images_filepath = join(config.input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(config.input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(config.input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(config.input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

def show_recon_grid(originals, reconstructions, labels, save_path=None):
    rows = len(originals)
    fig, axes = plt.subplots(rows, 2, figsize=(6, rows * 2.2))

    for i in range(rows):
        # Original
        axes[i, 0].imshow(originals[i], cmap='gray')
        axes[i, 0].set_title(f"Original [{labels[i]}]", fontsize=10, pad=4)
        axes[i, 0].axis('off')

        # Reconstruction
        axes[i, 1].imshow(reconstructions[i], cmap='gray')
        axes[i, 1].set_title(f"Reconstructed [{labels[i]}]", fontsize=10, pad=4)
        axes[i, 1].axis('off')

    plt.tight_layout(pad=1.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_reconstructed_images(model_output: torch.tensor, x_test: torch.tensor, y_test: torch.tensor):
    originals = []
    recons = []
    labels = []

    # reshape to (N, 28, 28)
    x_test = x_test.view(-1, 28, 28)
    model_output = model_output.view(-1, 28, 28).detach().numpy()

    # pick 10 random samples
    for _ in range(10):
        r = random.randint(0, len(x_test) - 1)

        originals.append(x_test[r])
        recons.append(model_output[r])
        labels.append(y_test[r].item())

    # use the new grid viewer
    show_recon_grid(
        originals,
        recons,
        labels,
        save_path="regenerated_images.png"
    )


if __name__ == "__main__":
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    x_test = x_test / 255.0
    x_test = x_test.view(x_test.shape[0], -1)

    mnsit_vae = VariationalAutoEncoder(
                    x_dim = config.input_dim,
                    hidden_enc = config.hidden_enc,
                    hidden_dec = config.hidden_dec,
                    latent_dim = config.latent_dim,
    )

    mnsit_vae.load_state_dict(torch.load("vae_weights.pth", weights_only = True))
    mnsit_vae.eval()

    model_output, _, _ = mnsit_vae(x_test)
    visualize_reconstructed_images(model_output, x_test, y_test)