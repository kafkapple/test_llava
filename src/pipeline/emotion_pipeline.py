from typing import Dict, Any, Optional, List
import torch
from PIL import Image
import numpy as np

class EmotionPipeline:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        """LLaVA 모델 로드"""
        if self.model_path is None:
            raise ValueError("model_path must be provided")
            
        from transformers import (
            LlavaNextVideoProcessor, 
            LlavaNextVideoForConditionalGeneration
        )
        
        processor = LlavaNextVideoProcessor.from_pretrained(
            self.model_path,
            load_in_4bit=True
        )
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        return {"processor": processor, "model": model}
    
    def _preprocess_image(self, image):
        """이미지 전처리"""
        if isinstance(image, np.ndarray):
            # OpenCV BGR to RGB
            if image.shape[2] == 3:
                image = image[..., ::-1].copy()
            # Convert to PIL Image
            image = Image.fromarray(image.astype('uint8'))
        return image
    
    def process(
        self,
        data: Dict[str, Any],
        id: Optional[str] = None
    ) -> Dict[str, Any]:
        """데이터 처리"""
        try:
            # 대 명확한 프롬프트 구성
            conversation = [
                {
                    "role": "system",
                    "content": "You are an emotion detection expert. Always respond with a single primary emotion word (like Happy, Sad, Angry, Surprised, Confused, Neutral) followed by a brief explanation. Format: [EMOTION]: explanation"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the primary emotion expressed in this content? Provide a single emotion word and brief explanation."},
                        {"type": "image" if "image" in data else "video"}
                    ]
                }
            ]
            
            # 프롬프트 적용
            prompt = self.model["processor"].apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            # 입력 처리
            if "image" in data:
                image = self._preprocess_image(data["image"])
                inputs = self.model["processor"](
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.model["model"].device)
            else:
                frames = [self._preprocess_image(frame) for frame in data["video"]]
                inputs = self.model["processor"](
                    text=prompt,
                    videos=frames,
                    return_tensors="pt",
                    padding=True
                ).to(self.model["model"].device)
            
            # 감정 예측
            with torch.inference_mode():
                output = self.model["model"].generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=3,
                    repetition_penalty=1.2
                )
            
            # 응답 처리 및 검증
            response = self.model["processor"].decode(
                output[0][2:],
                skip_special_tokens=True
            ).replace("ASSISTANT:", "").strip()
            
            # 응답에서 감정과 설명 추출
            if ":" in response:
                emotion, explanation = response.split(":", 1)
                emotion = emotion.strip().upper()
                explanation = explanation.strip()
            else:
                # 콜론이 없는 경우 전체 응답을 설명으로 처리
                emotion = "UNKNOWN"
                explanation = response.strip()
            
            # 기본 감정 목록과 대조
            valid_emotions = {
                "HAPPY", "SAD", "ANGRY", "SURPRISED", 
                "CONFUSED", "NEUTRAL", "FEARFUL", "DISGUSTED"
            }
            
            if emotion not in valid_emotions:
                emotion = "UNKNOWN"
            
            return {
                "id": id or "0",
                "emotion": emotion,
                "explanation": explanation,
                "success": True
            }
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return {
                "id": id or "0",
                "emotion": "ERROR",
                "explanation": str(e),
                "success": False
            }
    