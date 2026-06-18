import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor
)

# ==================================================
# CONFIGURAÇÕES
# ==================================================

torch_dtype=torch.float16
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

image_folder = "imagens"
output_folder = "imagens_llama"

os.makedirs(output_folder, exist_ok=True)

# ==================================================
# CARREGAR MODELO
# ==================================================

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

model.eval()

# ==================================================
# CARREGAR PROMPT
# ==================================================

with open(
    "prompt_sem_depoimento.txt",
    "r",
    encoding="utf-8"
) as f:
    prompt_text = f.read().strip()

# ==================================================
# PROCESSAR IMAGENS
# ==================================================

for filename in os.listdir(image_folder):

    if not filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp")
    ):
        continue

    path = os.path.join(image_folder, filename)

    image = Image.open(path).convert("RGB")

    # ----------------------------------------------
    # Prompt multimodal
    # ----------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # ----------------------------------------------
    # Geração
    # ----------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    # ----------------------------------------------
    # Remover prompt da saída
    # ----------------------------------------------

    generated_ids_trimmed = generated_ids[
        :,
        inputs.input_ids.shape[1]:
    ]

    generated_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # ----------------------------------------------
    # Exibir resultado
    # ----------------------------------------------

    print(f"\n{filename}")
    print(generated_text)

    # ----------------------------------------------
    # Exportar imagem legendada
    # ----------------------------------------------

    plt.figure(figsize=(10, 10))

    plt.imshow(image)

    plt.axis("off")

    plt.title(
        generated_text,
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

    print(f"✔ Salvo em: {output_path}")

print("\nProcessamento concluído.")