import requests
from PIL import Image
import torch

class LLaVAInference:
    def __init__(self, model, processor):
        """
        Initialize inference class
        """
        self.model = model
        self.processor = processor
        
    def load_image_from_url(self, image_url: str):
        """
        Load and preprocess image from URL
        """
        # Download image
        image = Image.open(requests.get(image_url, stream=True).raw)
        return image
        
    def generate_response(self, image, prompt: str):
        """
        Generate response for image and prompt
        """
        # Prepare inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200)
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response 