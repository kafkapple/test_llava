import os
from typing import Union, List
from pathlib import Path
import torch
from PIL import Image

from src.model_loader import LLaVALoader

class EmotionRecognizer:
    def __init__(self, cache_dir: Union[str, Path] = "./models"):
        self.cache_dir = Path(cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 모델 경로 설정
        self.model_paths = {
            "qwen-7b": "lmms-lab/llava-next-interleave-qwen-7b",
            "llama-8b": "lmms-lab/llama3-llava-next-8b"
        }
        
        self.models = {}
        self.processors = {}
        
        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 로드
        self._load_models()

    def _load_models(self):
        """모델과 토로세서를 로드하거나 다운로드합니다."""
        for model_name, model_path in self.model_paths.items():
            model_cache = self.cache_dir / model_name
            
            loader = LLaVALoader(model_name=model_path, local_dir=str(model_cache))
            model, processor = loader.load_model()
            
            self.models[model_name] = model
            self.processors[model_name] = processor

    def analyze_emotion(
        self,
        image: Union[str, Image.Image],
        text: str,
        model_name: str = "qwen-7b"
    ) -> dict:
        """이미지와 텍스트를 기반으로 감정을 분석합니다."""
        
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image)
        
        # 모델과 토로세서 선택
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
            
        model = self.models[model_name]
        processor = self.processors[model_name]
        
        # 프롬프트 구성
        prompt = f"""Analyze the emotion in this image and text:
Text: {text}
Please describe the emotional content and provide:
1. Primary emotion
2. Confidence level (0-100)
3. Supporting details from both image and text
"""

        # 입력 처리 및 모델 추론
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "model_used": model_name,
            "analysis": response,
            "raw_text": text
        }

    def batch_analyze(
        self,
        items: List[dict],
        model_name: str = "qwen-7b"
    ) -> List[dict]:
        """여러 이미지와 텍스트 쌍을 일괄 처리합니다."""
        results = []
        for item in items:
            result = self.analyze_emotion(
                image=item["image"],
                text=item["text"],
                model_name=model_name
            )
            results.append(result)
        return results

# 감정 인식기 초기화
recognizer = EmotionRecognizer(cache_dir="e:/models")

# 단일 분석
result = recognizer.analyze_emotion(
    image="E:\data\emotion\happy\happy_Image_2.jpg",
    text="Some text describing the situation",
    model_name="qwen-7b"  # or "llama-8b"
)

print(result)

# # 배치 분석
# batch_items = [
#     {"image": "image1.jpg", "text": "text1"},
#     {"image": "image2.jpg", "text": "text2"}
# ]
# results = recognizer.batch_analyze(batch_items)