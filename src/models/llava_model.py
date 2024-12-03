import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from typing import Dict, Any
from .base import MultiModalModel

class LLaVAModel(MultiModalModel):
    def __init__(self, model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"):
        self.model_id = model_id
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        
    def predict(self, inputs: Dict[str, Any]) -> str:
        """
        감정 예측 수행
        Args:
            inputs: {'text': str, 'video': video_array, 'image': image}
        """
        # 기본 프롬프트 설정
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the emotion in this? Answer in one word:"},
                {"type": next(iter(inputs.keys()))}  # 첫 번째 모달리티 타입 사용
            ]
        }]
        
        # 입력 처리
        processed = self.processor(
            text=self.processor.apply_chat_template(conversation, add_generation_prompt=True),
            **inputs,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 생성
        output = self.model.generate(**processed, max_new_tokens=10)
        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def get_embedding(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """마지막 히든 스테이트를 임베딩으로 사용"""
        processed = self.processor(**inputs, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**processed, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1) 