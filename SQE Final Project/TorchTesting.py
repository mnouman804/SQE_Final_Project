import requests
import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
import torch.testing as tt

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def load_image(url):
    try:
        return Image.open(requests.get(url, stream=True).raw).convert('RGB')
    except UnidentifiedImageError:
        return None

# Test Functionality
def test_image_captioning():
    test_images = [
        'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg',
        'https://via.placeholder.com/150',  # Low resolution test
        'https://www.hq.nasa.gov/alsj/a17/A17_FlightPlan.pdf',  # Non-image test
    ]

    for url in test_images:
        image = load_image(url)
        if image is None:
            print(f"Skipping non-image file: {url}")
            continue

        inputs = processor(image, return_tensors="pt")
        output_ids = model.generate(**inputs)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        print(f"Caption for {url}: {caption}")

        # Functional check: Ensure output is a non-empty string
        tt.assert_close(len(caption) > 0, True, msg=f"Caption empty for {url}")


# Performance test 
def test_inference_time():
    image = load_image('https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg')
    inputs = processor(image, return_tensors="pt")

    start_time = time.time()
    with torch.no_grad():
        model.generate(**inputs)
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Start time: {start_time:.2f} ms")
    print(f"End Time: {end_time:.2f} ms")
    print(f"Inference time: {inference_time:.2f} ms")


# Run tests
test_image_captioning()
test_inference_time()
