import requests
from PIL import Image
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from deepchecks.tabular.suites import data_integrity  # Correct import path for DataIntegrity check
from deepchecks.tabular import Dataset

# Load the BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Fetch and prepare the image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'  # Image URL
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")
out = model.generate(**inputs)
caption_conditional = processor.decode(out[0], skip_special_tokens=True)

# Unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)
caption_unconditional = processor.decode(out[0], skip_special_tokens=True)

# Display the results
print("Conditional Caption: ", caption_conditional)
print("Unconditional Caption: ", caption_unconditional)

# Prepare data for Deepchecks
test_data = {
    'image_url': [img_url],  # Image URL
    'conditional_caption': [caption_conditional],
    'unconditional_caption': [caption_unconditional]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(test_data)

# Create the Deepchecks dataset
dataset = Dataset(df)

# Perform data integrity check using Deepchecks
check_result = data_integrity().run(dataset)  # Run data integrity check

# Show results of the data integrity check
check_result.show()
check_result.save_as_html('deepchecks_advanced_reporter.html')



#Deepchecks Testing in This Code:
#Test Type: Data Integrity Testing
#The Deepchecks suite (data_integrity) checks the quality and consistency of the data in the provided DataFrame. The checks could include:
#Missing Values: Checking if any columns have missing or null values.
##Data Types: Ensuring that all columns have consistent data types.
#Duplicates: Detecting duplicate rows in the dataset.
#Constant Features: Identifying any columns that have the same value across all rows, which can indicate redundancy.