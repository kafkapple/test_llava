# Model loader class for different LLaVA models
from pathlib import Path
import os
from huggingface_hub import snapshot_download

class LLaVALoader:
    def __init__(self, model_name: str, model_type: str = "llama3", local_dir: str = "./models"):
        """
        Initialize LLaVA model loader
        model_name: HuggingFace model name (e.g., "lmms-lab/llama3-llava-next-8b")
        model_type: "llama3" or "qwen"
        local_dir: local directory to store/load models
        """
        self.model_name = model_name
        self.model_type = model_type
        self.local_dir = Path(local_dir)
        self.model_dir = self.local_dir / model_name.split('/')[-1]
        self.model = None
        self.processor = None
    
    def is_model_exists(self):
        """
        Check if model files exist in local directory
        """
        # 필수 파일들이 존재하는지 확인
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        return all((self.model_dir / file).exists() for file in required_files)
    
    def download_model(self):
        """
        Download model from HuggingFace Hub
        """
        print(f"Downloading model to: {self.model_dir}")
        try:
            model_path = snapshot_download(
                repo_id=self.model_name,
                local_dir=str(self.model_dir),
                local_dir_use_symlinks=False
            )
            print("Model downloaded successfully")
            return model_path
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
        
    def load_model(self):
        """
        Load model from local directory or download if not exists
        """
        # Create base directory if not exists
        os.makedirs(self.local_dir, exist_ok=True)
        
        # Check if model exists locally
        if self.is_model_exists():
            print(f"Loading model from local directory: {self.model_dir}")
            model_path = str(self.model_dir)
        else:
            print("Model not found locally. Downloading...")
            model_path = self.download_model()
            
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            
            print("Loading model into memory...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path
            )
            
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_path
            )
            
            print("Model and processor loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        return self.model, self.processor 