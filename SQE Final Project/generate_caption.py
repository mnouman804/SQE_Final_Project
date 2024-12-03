import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize the processor and model
def initialize_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# Function to generate caption from an image
def generate_caption(img_url: str, text: str = "a photography of"):
    processor, model = initialize_model()
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, text, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Function to generate a caption without any specific text
def generate_caption_no_text(img_url: str):
    processor, model = initialize_model()
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
