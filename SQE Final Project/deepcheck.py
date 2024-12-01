from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import deepchecks as dc

# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Path to your local image
image_path = '/Users/user/Pictures/Personal Pictures/result_sr.png'  # Update with the correct path
raw_image = Image.open(image_path).convert('RGB')

# Prepare the inputs for the model
text = "a photograph of"
inputs = processor(raw_image, text, return_tensors="pt")
inputs_unconditional = processor(raw_image, return_tensors="pt")

# Generate captions
out_conditional = model.generate(**inputs)
out_unconditional = model.generate(**inputs_unconditional)

# Decode the output to text
caption_conditional = processor.decode(out_conditional[0], skip_special_tokens=True)
caption_unconditional = processor.decode(out_unconditional[0], skip_special_tokens=True)

# Custom check for image captioning
def captioning_test():
    assert len(caption_conditional) > 10, f"Conditional caption is too short: {caption_conditional}"
    assert len(caption_unconditional) > 10, f"Unconditional caption is too short: {caption_unconditional}"
    assert isinstance(caption_conditional, str) and caption_conditional.strip() != "", "Conditional caption should be a non-empty string"
    assert isinstance(caption_unconditional, str) and caption_unconditional.strip() != "", "Unconditional caption should be a non-empty string"

# Run the test
captioning_test()

# Print the results
print(f"Conditional Caption: {caption_conditional}")
print(f"Unconditional Caption: {caption_unconditional}")
