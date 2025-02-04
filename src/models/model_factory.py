import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import logging
from pathlib import Path
from dotenv import load_dotenv
import torch
from huggingface_hub import snapshot_download
from typing import Optional, Dict, Any

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

class ModelManager:
    def __init__(self, token: str):
        self.token = token
        
    def ensure_model(
        self, 
        model_id: str, 
        save_directory: str,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        모델을 다운로드하거나 기존 모델을 로드합니다.
        
        Args:
            model_id (str): Hugging Face 모델 ID
            save_directory (str): 모델 저장 경로
            model_kwargs (dict, optional): 모델 로드 시 추가 매개변수
            
        Returns:
            Path: 모델이 저장된 경로
        """
        try:
            # 기본 저장 경로 설정
            save_path = Path(save_directory) / Path(model_id.split("/")[-1])
            save_path.mkdir(parents=True, exist_ok=True)
        
            
            if not save_path.exists() or not any(save_path.iterdir()):
                print(f"모델 '{model_id}' 다운로드 중... 경로: {save_path}")
                
                # 기본 모델 파일 다운로드
                snapshot_download(
                    repo_id=model_id,
                    local_dir=save_path,
                    token=self.token,
                    ignore_patterns=["*.md", "*.txt"]
                )
                
                # 추가 모델 컴포넌트 다운로드 및 저장
                model_kwargs = model_kwargs or {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "offload_folder": save_path / "offload"
                }
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    use_auth_token=self.token,
                    **model_kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, 
                    use_auth_token=self.token
                )
                config = AutoConfig.from_pretrained(
                    model_id, 
                    use_auth_token=self.token
                )
                
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                config.save_pretrained(save_path)
                
                print(f"모델 '{model_id}' 다운로드 완료")
            else:
                print(f"기존 모델 사용: {save_path}")
                
            return save_path
            
        except Exception as e:
            raise RuntimeError(f"모델 '{model_id}' 다운로드/로드 실패: {str(e)}")

