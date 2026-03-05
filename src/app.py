from flask import Flask, request, jsonify, render_template
import torch
from .config import Config
from .model import VariationalAutoEncoder
import io
import base64
from PIL import Image
import numpy as np


app = Flask(__name__)

config = Config()
model = VariationalAutoEncoder(
                    x_dim = config.input_dim,
                    hidden_enc = config.hidden_enc,
                    hidden_dec = config.hidden_dec,
                    latent_dim = config.latent_dim,
    )
model.load_state_dict(torch.load("vae_weights.pth", weights_only = True))
model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate_img", methods = ["GET"])
def generate_image():
    global config, model

    with torch.no_grad():
        z = torch.randn(1, config.latent_dim)
        x_tilde = model.decoder(z)
    x_tilde = x_tilde.detach().cpu().view(28, 28)
    
    img = x_tilde.numpy()

    # Normalize to 0–255
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    # Convert to PIL image
    image = Image.fromarray(img)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({
        "status": "success",
        "model": "vae",
        "img": img_base64,
        "latent_dim": config.latent_dim
    })



if __name__ == "__main__":
    app.run(debug = True)