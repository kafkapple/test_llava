import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.model_loader import LLaVALoader
from src.inference import LLaVAInference



def main():
    # Initialize model loader with specific local directory
    model_name = "lmms-lab/llama3-llava-next-8b"
    local_dir = "e:/models/llama3-llava-next-8b"
    
    loader = LLaVALoader(
        model_name=model_name,
        model_type="llama3",
        local_dir=local_dir
    )
    
    # Load model
    model, processor = loader.load_model()
    
    # Initialize inference
    inference = LLaVAInference(model, processor)
    
    # Test with sample image
    image_url = "E:/data/emotion/happy/happy_Image_2.jpg"
    prompt = "What can you see in this image?"
    
    # Load image and generate response
    image = inference.load_image_from_url(image_url)
    response = inference.generate_response(image, prompt)
    
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 