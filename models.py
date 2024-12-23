import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import logging
from pathlib import Path
from dotenv import load_dotenv
import torch

def download_huggingface_model(model_id: str, save_directory: str):
    """

    Args:
        model_id (str): Hugging Face 
        save_directory (str): 
    """
    try:
        logging.set_verbosity_info()
        logger = logging.get_logger("transformers")

        load_dotenv()
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN is not set in .env file.")

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)  

        # device_map offload
        print(f"Downloading model '{model_id}' to '{save_path}'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_auth_token=token,
            torch_dtype=torch.float16,  # torch.float16
            device_map="auto", 
            offload_folder=save_path / "offload"  # 일부 데이터를 디스크로 오프로드
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
        config = AutoConfig.from_pretrained(model_id, use_auth_token=token)

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        config.save_pretrained(save_path)

        print(f"Model '{model_id}' downloaded successfully and saved to '{save_path}'.")

    except Exception as e:
        print(f"Error downloading model '{model_id}': {e}")
if __name__ == "__main__":
    model_id = "meta-llama/Llama-2-7b-chat-hf"##"TencentGameMate/chinese-hubert-large" #input("Enter Hugging Face model ID (e.g., bert-base-uncased): ").strip()
    save_directory = "/home/joon/models"#input("Enter local directory to save the model: ").strip()

    download_huggingface_model(model_id, save_directory)
#pip install "transformers==4.30.0" "tokenizers<0.14"
