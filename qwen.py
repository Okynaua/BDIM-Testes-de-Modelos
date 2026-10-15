import os
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image

import torch
import matplotlib.pyplot as plt

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
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


# ==========================================
# CONFIGURAÇÕES
# ==========================================

model_name = "Qwen/Qwen3-VL-4B-Instruct"

output_folder = "imagens_qwen_sem_depoimento"

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# CARREGAR MODELO
# ==========================================

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name)

model.eval()

# ==========================================
# CARREGAR PROMPT
# ==========================================

with open(
    "prompt_sem_depoimento.txt",
    "r",
    encoding="utf-8"
) as f:

    prompt_text = f.read().strip()

# ==========================================
# LER URLS
# ==========================================

with open(
    "urls.txt",
    "r",
    encoding="utf-8"
) as f:

    urls = [linha.strip() for linha in f if linha.strip()]

# ==========================================
# PROCESSAR URLS
# ==========================================

for indice, url in enumerate(urls, start=1):

    try:

        print(f"\nProcessando: {url}")

        img_url = extrai_imagem_url(url)

        response = requests.get(
            img_url,
            timeout=30
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        # prompt multimodal
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )

        inputs = inputs.to(model.device)

        with torch.no_grad():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        generated_ids_trimmed = [

            output_ids[len(input_ids):]

            for input_ids, output_ids in zip(
                inputs.input_ids,
                generated_ids
            )
        ]

        caption = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # ==========================================
        # EXPORTAR IMAGEM COM LEGENDA
        # ==========================================

        plt.figure(figsize=(10, 10))

        plt.imshow(image)

        plt.axis("off")

        plt.title(
            caption,
            fontsize=10,
            wrap=True
        )

        output_path = os.path.join(
            output_folder,
            f"historia_{indice}_caption.png"
        )

        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=200
        )

        plt.close()

        print(f"Imagem: {img_url}")
        print(f"Resultado:\n{caption}")
        print(f"Salvo em: {output_path}")

    except Exception as e:

        print(
            f"Erro ao processar {url}: {e}"
        )

print("\nProcessamento concluído.")