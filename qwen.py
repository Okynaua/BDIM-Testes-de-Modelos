import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# CONFIGURAÇÕES
model_name = "Qwen/Qwen3-VL-4B-Instruct"

image_folder = "imagens"
output_folder = "imagens_qwen"

os.makedirs(output_folder, exist_ok=True)

# CARREGAR MODELO
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name)

model.eval()

# CARREGAR PROMPT
with open("prompt_sem_depoimento.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read().strip()

# PROCESSAR IMAGENS
for filename in os.listdir(image_folder):

    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):

        path = os.path.join(image_folder, filename)

        # carregar imagem
        image = Image.open(path).convert("RGB")

        # prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # formatar prompt
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # preparar inputs
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )

        inputs = inputs.to(model.device)

        # gerar descrição
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        # remover tokens do prompt
        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        caption = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    
        # EXPORTAR COM MATPLOTLIB
    

        plt.figure(figsize=(10, 10))

        plt.imshow(image)

        plt.axis("off")

        plt.title(
            caption,
            fontsize=10,
            wrap=True
        )

        # salvar imagem com legenda
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