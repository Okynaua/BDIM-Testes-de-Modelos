import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# CONFIGURAÇÕES
model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

image_folder = "imagens"
output_folder = "imagens_smolvlm"

os.makedirs(output_folder, exist_ok=True)

# CARREGAR MODELO
processor = AutoProcessor.from_pretrained(model_name)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
).to("cuda")

model.eval()

# CARREGAR PROMPT
with open("prompt_sem_depoimento.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read().strip()

# PROCESSAR IMAGENS
for filename in os.listdir(image_folder):

    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):

        path = os.path.join(image_folder, filename)

        image = Image.open(path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
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

        # remover o prompt da saída
        if "Assistant:" in generated_text:
            caption = generated_text.split("Assistant:")[-1].strip()
        else:
            caption = generated_text.strip()

        # EXPORTAR COM MATPLOTLIB

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
            f"{os.path.splitext(filename)[0]}_caption.png"
        )

        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=200
        )

        plt.close()

        print(f"\n{filename}")
        print(caption)
        print(f"Salvo em: {output_path}")

print("\nProcessamento concluído.")