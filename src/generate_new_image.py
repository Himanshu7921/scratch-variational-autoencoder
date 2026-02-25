import matplotlib.pyplot as plt
import torch
from .model import VariationalAutoEncoder
from .config import Config

config = Config()
def generate_new_images(model: VariationalAutoEncoder, n_images: int = 16):
    latent_dim = model.encoder.linear_layer_2.weight.shape[1]

    rows = cols = int(n_images ** 0.5)

    fig, ax = plt.subplots(rows, cols, figsize=(6, 6))

    for r in range(rows):
        for c in range(cols):
            z = torch.randn(1, latent_dim)

            x_tilde = model.decoder(z)

            img = x_tilde.detach().cpu().view(28, 28)

            ax[r, c].imshow(img, cmap="gray")
            ax[r, c].axis("off")

    plt.show()

if __name__ == "__main__":
    mnsit_vae = VariationalAutoEncoder(
                    x_dim = config.input_dim,
                    hidden_enc = config.hidden_enc,
                    hidden_dec = config.hidden_dec,
                    latent_dim = config.latent_dim,
    )

    mnsit_vae.load_state_dict(torch.load("vae_weights.pth", weights_only = True))
    mnsit_vae.eval()
    generate_new_images(mnsit_vae, n_images = 300)