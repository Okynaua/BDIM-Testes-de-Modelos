import os
from PIL import Image
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel

# carregar modelo
model = VisionEncoderDecoderModel.from_pretrained("laicsiifes/swin-distilbertimbau")
tokenizer = AutoTokenizer.from_pretrained("laicsiifes/swin-distilbertimbau")
image_processor = AutoImageProcessor.from_pretrained("laicsiifes/swin-distilbertimbau")

model.eval()

image_folder = "imagens"
output_folder = "imagens_swin-distilbertimbau"

os.makedirs(output_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for filename in os.listdir(image_folder):

    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):

        path = os.path.join(image_folder, filename)

        image = Image.open(path).convert("RGB")

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
            f"{os.path.splitext(filename)[0]}_caption.png"
        )

        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300
        )

        print(f"\n{filename}")
        print(caption)
        print(f"Salvo em: {output_path}")

        plt.close()

print("\nProcessamento concluído.")