import os
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image

import torch
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    VisionEncoderDecoderModel
)


def extrai_imagem_url(url):

    page = requests.get(url, timeout=30)
    page.raise_for_status()

    soup = BeautifulSoup(page.content, "html.parser")

    div_img = soup.find("div", class_="col pr-lg-0 py-4")

    if div_img is None:
        raise Exception("Div da imagem não encontrada")

    img = div_img.find("img")

    if img is None:
        raise Exception("Tag <img> não encontrada")

    return img["src"]


# ==========================
# Carrega modelo
# ==========================

model = VisionEncoderDecoderModel.from_pretrained(
    "laicsiifes/swin-distilbertimbau"
)

tokenizer = AutoTokenizer.from_pretrained(
    "laicsiifes/swin-distilbertimbau"
)

image_processor = AutoImageProcessor.from_pretrained(
    "laicsiifes/swin-distilbertimbau"
)

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

output_folder = "imagens_swin_distilbertimbau"
os.makedirs(output_folder, exist_ok=True)

# ==========================
# Lê urls.txt
# ==========================

with open("urls.txt", "r", encoding="utf-8") as f:
    urls = [linha.strip() for linha in f if linha.strip()]

# ==========================
# Processa cada URL
# ==========================

for indice, url in enumerate(urls, start=1):

    try:

        print(f"\nProcessando: {url}")

        img_url = extrai_imagem_url(url)

        response = requests.get(img_url, timeout=30)
        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        pixel_values = image_processor(
            image,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=60,
                num_beams=5,
                early_stopping=True
            )

        caption = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(caption, wrap=True)

        output_path = os.path.join(
            output_folder,
            f"historia_{indice}_caption.png"
        )

        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300
        )

        plt.close()

        print(f"Imagem: {img_url}")
        print(f"Legenda: {caption}")
        print(f"Salvo em: {output_path}")

    except Exception as e:

        print(
            f"Erro ao processar {url}: {e}"
        )

print("\nProcessamento concluído.")