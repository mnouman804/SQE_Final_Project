import requests
from PIL import Image
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, model_evaluation
from deepchecks.core.checks import CustomCheck
from deepchecks.core.suite import Suite
from deepchecks.core import CheckResult

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Fetch and prepare the image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")
conditional_output = model.generate(**inputs)
caption_conditional = processor.decode(conditional_output[0], skip_special_tokens=True)

# Unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")
unconditional_output = model.generate(**inputs)
caption_unconditional = processor.decode(unconditional_output[0], skip_special_tokens=True)

# Display results
print("Conditional Caption:", caption_conditional)
print("Unconditional Caption:", caption_unconditional)

# Prepare DataFrame for evaluation
data = {
    'image_url': [img_url],
    'conditional_caption': [caption_conditional],
    'unconditional_caption': [caption_unconditional]
}
df = pd.DataFrame(data)

# Create a Deepchecks Dataset
dataset = Dataset(df)

# Run Deepchecks Data Integrity Suite
integrity_suite = data_integrity()
integrity_results = integrity_suite.run(dataset)
integrity_results.show()

# Save report as HTML
integrity_results.save_as_html('deepchecks_integrity_report.html')

# Custom Evaluation: Caption Length Check
class CaptionLengthCheck(CustomCheck):
    def run(self, dataset: Dataset) -> CheckResult:
        caption_lengths = dataset.data[['conditional_caption', 'unconditional_caption']].applymap(len)
        return CheckResult(
            value=caption_lengths.describe(), 
            display=["table"]
        )

# Define a custom suite including caption length check
custom_suite = Suite("Custom Caption Evaluation Suite", checks=[
    CaptionLengthCheck('Caption Length Distribution')
])

# Run the custom suite
custom_results = custom_suite.run(dataset)
custom_results.show()
custom_results.save_as_html('custom_caption_evaluation.html')
