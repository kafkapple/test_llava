from typing import Dict, Any, Optional, List
import torch
from PIL import Image
import numpy as np

class EmotionPipeline:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.image_size = 336
        self.patch_size = 14
        self.num_frames = 8
        self.hidden_size = 1024
        self.max_sequence_length = 256
        self.prompt_version = 1  # 기본값 1
        
        # 생성 설정 추가
        self.generation_config = {
            "max_new_tokens": 256,
            "do_sample": True,
            "num_beams": 5,  # 빔 서치
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,  # top-k 샘플링
            "repetition_penalty": 1.2,
            "length_penalty": 1.0,  # 길이에 대한 페널티
            "no_repeat_ngram_size": 3,  # n-gram 반복 방지
            "num_return_sequences": 1,
            "early_stopping": True  # 빔 서치 조기 종료
        }
        
        self.model = self._load_model()
        
    def _load_model(self):
        """LLaVA 모델 로드"""
        if self.model_path is None:
            raise ValueError("model_path must be provided")
            
        from transformers import (
            LlavaNextVideoProcessor, 
            LlavaNextVideoForConditionalGeneration,
            LlavaNextVideoConfig,
            BitsAndBytesConfig
        )
        
        try:
            print("Loading model...")
            
            # 1. 메모리 최적화 설정
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # 2. 모델 설정 준비
            model_config = LlavaNextVideoConfig.from_pretrained(self.model_path)
            
            # 3. 비전 설정 업데이트
            model_config.vision_config.image_size = self.image_size
            model_config.vision_config.patch_size = self.patch_size
            model_config.vision_config.num_channels = 3
            model_config.vision_config.num_frames = self.num_frames
            model_config.vision_config.hidden_size = self.hidden_size
            model_config.vision_config.intermediate_size = 4 * self.hidden_size
            model_config.vision_config.num_hidden_layers = 24
            model_config.vision_config.num_attention_heads = 16
            model_config.vision_config.vision_feature_select_strategy = "full"
            
            # 4. 모델 로드
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_path,
                config=model_config,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # 5. 프로세서 로드
            processor = LlavaNextVideoProcessor.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                config=model_config
            )
            
            # 6. 프로세서 설정 업데이트
            processor.image_processor.size = {
                "height": self.image_size,
                "width": self.image_size
            }
            processor.image_processor.patch_size = {
                "height": self.patch_size,
                "width": self.patch_size
            }
            processor.image_processor.do_resize = True
            processor.image_processor.size = self.image_size
            processor.image_processor.do_normalize = True
            processor.image_processor.image_mean = [0.485, 0.456, 0.406]
            processor.image_processor.image_std = [0.229, 0.224, 0.225]
            
            # 추가: 명시적 프로세서 설정
            processor.patch_size = self.patch_size
            processor.vision_feature_select_strategy = "full"
            processor.config = model_config
            
            # 7. 토크나이저 설정
            video_token = "<video>"
            if video_token not in processor.tokenizer.get_vocab():
                special_tokens = {
                    "additional_special_tokens": [video_token],
                    "pad_token": processor.tokenizer.eos_token,
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                }
                processor.tokenizer.add_special_tokens(special_tokens)
                # 토크나이저 설정 후 모델 임베딩 리사이즈
                model.resize_token_embeddings(len(processor.tokenizer))
            
            processor.tokenizer.padding_side = "left"
            video_token_id = processor.tokenizer.convert_tokens_to_ids(video_token)
            
            # 8. 모델 관련 설정 업데이트
            model.config.video_token = video_token
            model.config.video_token_index = video_token_id
            model.config.use_video_tokens = True
            
            # video_token을 클래스 변수로 저장
            self.video_token = video_token
            
            # 9. GPU로 이동
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            print("\nModel loaded successfully")
            self.device = device  # 디바이스 정보 저장
            return {
                "processor": processor,
                "model": model,
                "device": device
            }
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _preprocess_image(self, image):
        """이미지 전처리"""
        if isinstance(image, np.ndarray):
            # OpenCV BGR to RGB
            if image.shape[2] == 3:
                image = image[..., ::-1].copy()
            # Convert to PIL Image
            image = Image.fromarray(image.astype('uint8'))
        
        # 이미지 크기 조정
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.LANCZOS
        )
        return image
    
    def _preprocess_video(self, frames: List[np.ndarray], max_frames: int = 8) -> List[Image.Image]:
        """비디오 프레임 전처리"""
        print("=== Video Preprocessing ===")
        print(f"Input frames: shape={frames[0].shape}, count={len(frames)}")
        
        # 균일한 간격으로 프레임 샘플링
        total_frames = len(frames)
        if total_frames > max_frames:
            indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            print(f"Sampled frames: count={len(frames)}, indices={indices}")
        elif total_frames < max_frames:
            # 마지막 프레임으로 패딩
            frames = frames + [frames[-1]] * (max_frames - total_frames)
            print(f"Padded frames: count={len(frames)}")
        
        # 각 프레임 전처리
        processed_frames = []
        for frame in frames:
            # BGR to RGB 변환
            if frame.shape[2] == 3:
                frame = frame[..., ::-1].copy()
            # 이미지 크기 표준화
            frame = Image.fromarray(frame.astype('uint8'))
            frame = frame.resize(
                (336, 336),  # 336으로 변경
                Image.Resampling.LANCZOS
            )
            processed_frames.append(frame)
            
        print(f"Processed frames: size={processed_frames[0].size}, count={len(processed_frames)}")
        return processed_frames
    
    def _preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """프레임을 모델이 기대하는 형태로 변환"""
        # 1. 이미지를 텐서로 변환
        video_tensors = []
        for frame in frames:
            # numpy 배열로 변환
            frame_array = np.array(frame)
            frame_array = frame_array.astype(np.float32) / 255.0
            
            # 채널 차원을 앞으로 이동 (H, W, C) -> (C, H, W)
            frame_array = np.transpose(frame_array, (2, 0, 1))
            
            # 정규화 (ImageNet 평균/표준편차)
            mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            frame_array = (frame_array - mean) / std
            
            video_tensors.append(frame_array)
        
        # 2. 비디오 텐서 생성 (F, C, H, W)
        video_tensor = np.stack(video_tensors)
        
        # 3. 배치 차원 추가 (B, F, C, H, W)
        video_tensor = np.expand_dims(video_tensor, 0)
        
        # 4. PyTorch 텐서로 변환
        video_tensor = torch.from_numpy(video_tensor).to(torch.float16)
        
        return video_tensor
    
    def process(self, data: Dict[str, Any], id: Optional[str] = None) -> Dict[str, Any]:
        try:
            # 1. 입력 데이터 전처리
            if "video" in data:
                frames = self._preprocess_video(data["video"])
                input_type = "video"
            else:
                image = self._preprocess_image(data["image"])
                frames = [image] * self.num_frames
                input_type = "image"
            
            # 2. 프롬프트 구성
            if self.prompt_version == 1:
                base_prompt = "Describe the emotion in this content."
            else:
                base_prompt = (
                    "What is the emotion in this content? "
                    "Use ONLY these labels: HAPPY, SAD, ANGRY, SURPRISED, "
                    "CONFUSED, NEUTRAL, FEARFUL, DISGUSTED."
                )
            
            # 3. 비디오 프레임 처리
            video_tensor = self._preprocess_frames(frames)
            
            # 4. 텍스트 처리
            prompt = f"{self.video_token} {base_prompt}"
            text_inputs = self.model["processor"].tokenizer(
                [prompt],
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length
            )
            
            # 5. 입력 구성
            inputs = {
                "input_ids": text_inputs.input_ids.to(self.device),
                "attention_mask": text_inputs.attention_mask.to(self.device),
                "pixel_values_videos": video_tensor.to(self.device)
            }
            
            # 6. 디바이스 이동
            inputs = {
                k: v.to(self.device) 
                for k, v in inputs.items()
            }
            
            # 7. 디버그 정보 출력
            print("\n=== Input Debug Info ===")
            print(f"Input type: {input_type}")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k} shape: {v.shape}")
            
            print("\n=== Model Config ===")
            print(f"Patch size: {self.model['model'].config.vision_config.patch_size}")
            print(f"Vision strategy: {self.model['model'].config.vision_config.vision_feature_select_strategy}")
            print(f"Image size: {self.model['model'].config.vision_config.image_size}")
            
            # 8. 생성 설정
            generation_config = self.generation_config.copy()
            
            # 9. 생성 실행
            with torch.inference_mode():
                print("\n=== Generation Start ===")
                print("\n=== Generation Config ===")
                for k, v in generation_config.items():
                    print(f"{k}: {v}")
                    
                output = self.model["model"].generate(
                    **inputs,
                    **generation_config
                )
                print(f"Output shape: {output.shape}")
                print(f"Raw output: {output.tolist()}")
            
            # 10. 응답 처리
            print("\n=== Response Processing ===")
            response = self.model["processor"].decode(
                output[0], 
                skip_special_tokens=True
            ).strip()
            print(f"Raw response: {response}")
            
            # 11. 응답 파싱
            if ":" in response:
                category, explanation = response.split(":", 1)
                category = category.strip().upper()
                if category.startswith("[") and category.endswith("]"):
                    category = category[1:-1]
                explanation = explanation.strip()
            else:
                category = "OBSERVATION"
                explanation = response.strip()
            
            valid_emotions = {
                "HAPPY", "SAD", "ANGRY", "SURPRISED", 
                "CONFUSED", "NEUTRAL", "FEARFUL", "DISGUSTED"
            }
            
            is_emotion = category in valid_emotions
            
            return {
                "id": id or "0",
                "emotion": category if is_emotion else "OBSERVATION",
                "explanation": explanation,
                "is_emotion": is_emotion,
                "success": True,
                "debug_info": {
                    "input_type": input_type,
                    "frame_count": len(frames),
                    "token_count": len(inputs["input_ids"][0])
                }
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Error during processing: {str(e)}\n"
            error_msg += traceback.format_exc()
            print(error_msg)
            
            return {
                "id": id or "0",
                "emotion": "ERROR",
                "explanation": str(e),
                "is_emotion": False,
                "success": False,
                "debug_info": {
                    "error_type": type(e).__name__,
                    "full_error": error_msg,
                    "input_type": "video" if "video" in data else "image"
                }
            }
    