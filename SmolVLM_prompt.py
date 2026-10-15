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


def extrai_historia(url):

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    div_img = soup.find(
        "div",
        class_="col pr-lg-0 py-4"
    )

    img = div_img.find("img")
    img_link = img["src"]

    texto_div = soup.find(
        "div",
        id="historia-texto"
    )

    full_text = texto_div.find(
        "div",
        class_="full-text"
    )

    paragrafos = full_text.find_all("p")

    texto = "\n".join(
        p.get_text(" ", strip=True)
        for p in paragrafos
    )

    return texto, img_link


# ==========================================
# CONFIGURAÇÕES
# ==========================================

model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

output_folder = "imagens_smolvlm_com_depoimento"

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# MODELO
# ==========================================

processor = AutoProcessor.from_pretrained(
    model_name
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
).to("cuda")

model.eval()

# ==========================================
# PROMPT
# ==========================================

with open(
    "prompt_com_depoimento.txt",
    "r",
    encoding="utf-8"
) as f:

    prompt_text = f.read().strip()

# ==========================================
# URLS
# ==========================================

with open(
    "urls.txt",
    "r",
    encoding="utf-8"
) as f:

    urls = [
        linha.strip()
        for linha in f
        if linha.strip()
    ]

# ==========================================
# PROCESSAMENTO
# ==========================================

for indice, url in enumerate(urls, start=1):

    try:

        texto_historia, img_url = extrai_historia(
            url
        )

        response = requests.get(img_url)
        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        prompt_completo = (
            f"{prompt_text}\n\n"
            f"=== DEPOIMENTO ===\n"
            f"{texto_historia}"
        )

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
                        "text": prompt_completo
                    }
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(model.device)
            if torch.is_tensor(v)
            else v
            for k, v in inputs.items()
        }

        with torch.no_grad():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        if "Assistant:" in generated_text:
            caption = generated_text.split(
                "Assistant:"
            )[-1].strip()
        else:
            caption = generated_text.strip()

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

        print(f"\nURL: {url}")
        print(caption)

    except Exception as e:

        print(
            f"Erro ao processar {url}: {e}"
        )

print("\nProcessamento concluído.")