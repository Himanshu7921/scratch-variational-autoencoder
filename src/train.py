import torch
import torch.nn.functional as F
from tqdm import tqdm
from os.path  import join
from torch.utils.data import DataLoader, TensorDataset

from .model import VariationalAutoEncoder
from .config import Config
from .data_loader import MnistDataloader
from .klloss import kl_divergence


config = Config()
training_images_filepath = join(config.input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(config.input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(config.input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(config.input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

def train_vae(model, x_train, optimizer, epochs=10, batch_size=128, device="cpu"):

    model.train()
    latent_dim = model.latent_dim
    model.to(device)

    # Create DataLoader
    dataset = TensorDataset(x_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, leave=True)

        for batch in loop:
            x = batch[0].to(device)

            # forward pass
            x_tilde, mu, log_var = model(x)

            # losses
            recon_loss = F.binary_cross_entropy(x_tilde, x, reduction="mean")
            kl_loss = kl_divergence(mu, log_var) / x.size(0)   # normalize per batch
            kl_loss = kl_loss / (batch_size * latent_dim)
            loss = recon_loss + kl_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tqdm status bar
            loop.set_description(f"Epoch {epoch}/{epochs}")
            loop.set_postfix(
                loss=float(loss),
                recon=float(recon_loss),
                kl=float(kl_loss)
            )

    print("Training complete.")


if __name__ == "__main__":
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    x_train = x_train / 255.0
    x_train = x_train.view(x_train.shape[0], -1)

    # Get the Model Ready
    mnsit_vae = VariationalAutoEncoder(
                    x_dim = config.input_dim,
                    hidden_enc = config.hidden_enc,
                    hidden_dec = config.hidden_dec,
                    latent_dim = config.latent_dim,
    )

    optimizer = torch.optim.Adam(params = mnsit_vae.parameters(), lr = config.lr)

    train_vae(
        auto_encoder = mnsit_vae,
        x = x_train,
        optimizer = optimizer,
        epochs = config.epochs,
        batch_size = config.batch_size
    )

    # Save the Model After Training
    torch.save(mnsit_vae.state_dict(), "vae_weights.pth")
    print("\nTraining completed. The model has been saved to 'autoencoder_weights.pth'.")