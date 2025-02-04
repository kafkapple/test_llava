import torch
from transformers import (
    LlavaNextVideoProcessor, 
    LlavaNextVideoForConditionalGeneration
)
from typing import Dict, Any, List
from .base import MultiModalModel


class LLaVAModel(MultiModalModel):
    def __init__(
        self, 
        model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
    ):
        self.model_id = model_id
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map='auto',
            load_in_4bit=True
        )
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        
    def predict(self, inputs: Dict[str, Any]) -> str:
        """감정 예측 수행
        Args:
            inputs: {'text': str, 'video': video_frames} or {'text': str, 'image': image}
        """
        # 기본 프롬프트 설정 
        prompt = "What is the emotion in this? Answer in one word:"
        
        # 입력 처리
        if 'video' in inputs:
            processed = self.processor(
                text=prompt,
                videos=inputs['video'],
                return_tensors="pt"
            ).to(self.model.device)
        else:
            # 이미지를 단일 프레임 비디오로 처리
            processed = self.processor(
                text=prompt,
                videos=[inputs['image']] * 8,  # 8 프레임으로 복제
                return_tensors="pt"
            ).to(self.model.device)
        
        # 생성
        output = self.model.generate(
            **processed,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def get_embedding(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """마지막 히든 스테이트를 임베딩으로 사용"""
        if 'video' in inputs:
            processed = self.processor(
                videos=inputs['video'],
                return_tensors="pt"
            ).to(self.model.device)
        else:
            processed = self.processor(
                videos=[inputs['image']] * 8,
                return_tensors="pt"
            ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**processed, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1) 