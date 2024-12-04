import requests
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchmetrics.text.bleu import BLEUScore
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Fetch and prepare the image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")
outputs = model.generate(**inputs)
caption_conditional = processor.decode(outputs[0], skip_special_tokens=True)

# Unconditional image captioning
inputs_uncond = processor(raw_image, return_tensors="pt")
outputs_uncond = model.generate(**inputs_uncond)
caption_unconditional = processor.decode(outputs_uncond[0], skip_special_tokens=True)

# Reference captions for BLEU evaluation
reference_captions = [["a photography of a cat", "a photo of a feline", "a picture of a cat"]]

# Evaluate with BLEU Score
bleu = BLEUScore(n_gram=4)
bleu_score_conditional = bleu([caption_conditional.split()], reference_captions)
bleu_score_unconditional = bleu([caption_unconditional.split()], reference_captions)

# Compute loss function
def compute_loss(model, inputs, captions):
    labels = processor(captions, return_tensors="pt", padding=True).input_ids
    outputs = model(**inputs, labels=labels)
    return outputs.loss

loss_conditional = compute_loss(model, inputs, [caption_conditional])
loss_unconditional = compute_loss(model, inputs_uncond, [caption_unconditional])

# Save data to a DataFrame
report_data = {
    "Caption Type": ["Conditional", "Unconditional"],
    "Generated Caption": [caption_conditional, caption_unconditional],
    "BLEU Score": [bleu_score_conditional.item(), bleu_score_unconditional.item()],
    "Loss": [loss_conditional.item(), loss_unconditional.item()],
}

df_report = pd.DataFrame(report_data)

# Save report as an HTML file
df_report.to_html("model_evaluation_report.html", index=False)

# Display the report in the console
print(df_report)

# Plot BLEU Scores and Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(report_data["Caption Type"], report_data["BLEU Score"], color='skyblue')
plt.title("BLEU Scores")
plt.ylabel("Score")

plt.subplot(1, 2, 2)
plt.bar(report_data["Caption Type"], report_data["Loss"], color='salmon')
plt.title("Loss Values")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("evaluation_metrics.png")
plt.show()
