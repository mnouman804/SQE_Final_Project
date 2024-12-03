import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import pytest
from generate_caption import generate_caption, generate_caption_no_text  # Import the functions from your code

# Test case for generating caption with custom text
def test_generate_caption_with_text():
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    text = "a photography of"
    caption = generate_caption(img_url, text)
    assert isinstance(caption, str), "The caption should be a string"
    assert len(caption) > 0, "The caption should not be empty"

# Test case for generating caption without custom text
def test_generate_caption_without_text():
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    caption = generate_caption_no_text(img_url)
    assert isinstance(caption, str), "The caption should be a string"
    assert len(caption) > 0, "The caption should not be empty"
