from typing import Dict, Any, Optional, List
import torch
from PIL import Image
import numpy as np
import re

class EmotionPipeline:
    def __init__(self, model_path: Optional[str] = None, seed: Optional[int] = None):
        self.model_path = model_path
        self.model = None
        self.image_size = 336
        self.patch_size = 14
        self.num_frames = 8
        self.hidden_size = 1024
        self.max_sequence_length = 256
        self.prompt_version = 1
        
        # 랜덤 시드 설정
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 생성 설정 수정
        self.generation_config = {
            "max_new_tokens": 512,  # 토큰 수 증가
            "do_sample": True,
            "num_beams": 3,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.3,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": 1,
            "early_stopping": True,
            "min_length": 100  # 최소 길이 증가
        }
        
        self.model = self._load_model()
        
        # video_token이 설정된 후에 bad_words_ids 추가
        self.generation_config["bad_words_ids"] = [[self.model["processor"].tokenizer.convert_tokens_to_ids(self.video_token)]]
        
    def _load_model(self):
        """Load LLaVA model"""
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
            
            # 1. Memory optimization settings
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # 2. Prepare model configuration
            model_config = LlavaNextVideoConfig.from_pretrained(self.model_path)
            
            # 3. Update vision configuration
            model_config.vision_config.image_size = self.image_size
            model_config.vision_config.patch_size = self.patch_size
            model_config.vision_config.num_channels = 3
            model_config.vision_config.num_frames = self.num_frames
            model_config.vision_config.hidden_size = self.hidden_size
            model_config.vision_config.intermediate_size = 4 * self.hidden_size
            model_config.vision_config.num_hidden_layers = 24
            model_config.vision_config.num_attention_heads = 16
            model_config.vision_config.vision_feature_select_strategy = "full"
            
            # 4. Load model
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_path,
                config=model_config,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # 5. Load processor
            processor = LlavaNextVideoProcessor.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                config=model_config
            )
            
            # 6. Update processor settings
            processor.config = model_config
            processor.patch_size = self.patch_size
            processor.vision_feature_select_strategy = "full"
            
            # Image processor settings
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
            
            # 7. Tokenizer settings
            video_token = "<video>"
            if video_token not in processor.tokenizer.get_vocab():
                special_tokens = {
                    "additional_special_tokens": [video_token],
                    "pad_token": processor.tokenizer.eos_token,
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                }
                processor.tokenizer.add_special_tokens(special_tokens)
                # Resize model embeddings after tokenizer settings
                model.resize_token_embeddings(len(processor.tokenizer))
            
            processor.tokenizer.padding_side = "left"
            video_token_id = processor.tokenizer.convert_tokens_to_ids(video_token)
            
            # 8. Update model settings
            model.config.video_token = video_token
            model.config.video_token_index = video_token_id
            model.config.use_video_tokens = True
            
            # Save video_token as class variable
            self.video_token = video_token
            
            # 9. Move to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            print("\nModel loaded successfully")
            self.device = device  # Save device information
            return {
                "processor": processor,
                "model": model,
                "device": device
            }
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _preprocess_image(self, image):
        """Preprocess image"""
        if isinstance(image, np.ndarray):
            # OpenCV BGR to RGB
            if image.shape[2] == 3:
                image = image[..., ::-1].copy()
            # Convert to PIL Image
            image = Image.fromarray(image.astype('uint8'))
        
        # Resize image
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.LANCZOS
        )
        return image
    
    def _preprocess_video(self, frames: List[np.ndarray], max_frames: int = 8) -> List[Image.Image]:
        """Preprocess video frames"""
        print("=== Video Preprocessing ===")
        print(f"Input frames: shape={frames[0].shape}, count={len(frames)}")
        
        total_frames = len(frames)
        
        if total_frames > max_frames:
            # Strategy 1: Include start, middle, and end frames
            selected_frames = []
            
            # Start frame
            selected_frames.append(frames[0])
            
            # End frame
            selected_frames.append(frames[-1])
            
            # Select remaining 6 frames from the middle
            if total_frames > 3:
                # Calculate frame differences (motion detection)
                frame_diffs = []
                for i in range(1, total_frames):
                    diff = np.mean(np.abs(frames[i].astype(np.float32) - frames[i-1].astype(np.float32)))
                    frame_diffs.append((i, diff))
                
                # Sort by difference in descending order
                frame_diffs.sort(key=lambda x: x[1], reverse=True)
                
                # Select top 6 frames with largest differences (maintain temporal order)
                important_indices = sorted([idx for idx, _ in frame_diffs[:6]])
                selected_frames.extend([frames[i] for i in important_indices])
            
            # Pad with last frame if needed
            while len(selected_frames) < max_frames:
                selected_frames.append(frames[-1])
            
            # Limit to max_frames
            frames = selected_frames[:max_frames]
            print(f"Selected frames using motion detection, count: {len(frames)}")
            
        elif total_frames < max_frames:
            # Pad with last frame if insufficient
            frames = frames + [frames[-1]] * (max_frames - total_frames)
            print(f"Padded frames: count={len(frames)}")
        
        # Preprocess each frame
        processed_frames = []
        for frame in frames:
            # BGR to RGB conversion
            if frame.shape[2] == 3:
                frame = frame[..., ::-1].copy()
            # Standardize image size
            frame = Image.fromarray(frame.astype('uint8'))
            frame = frame.resize(
                (self.image_size, self.image_size),
                Image.Resampling.LANCZOS
            )
            processed_frames.append(frame)
            
        print(f"Processed frames: size={processed_frames[0].size}, count={len(processed_frames)}")
        return processed_frames
    
    def _preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """Convert frames to model-expected format"""
        # 1. Convert images to tensors
        video_tensors = []
        for frame in frames:
            # Convert to numpy array
            frame_array = np.array(frame)
            frame_array = frame_array.astype(np.float32) / 255.0
            
            # Move channel dimension forward (H, W, C) -> (C, H, W)
            frame_array = np.transpose(frame_array, (2, 0, 1))
            
            # Normalize (ImageNet mean/std)
            mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            frame_array = (frame_array - mean) / std
            
            video_tensors.append(frame_array)
        
        # 2. Create video tensor (F, C, H, W)
        video_tensor = np.stack(video_tensors)
        
        # 3. Add batch dimension (B, F, C, H, W)
        video_tensor = np.expand_dims(video_tensor, 0)
        
        # 4. Convert to PyTorch tensor
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
                if input_type == "video":
                    base_prompt = (
                        "Analyze this video clip in detail:\n"
                        "1. Describe the visible emotions and expressions\n"
                        "2. Note any changes in emotional state\n"
                        "3. Describe relevant body language and gestures\n"
                        "4. Mention the setting and context if relevant\n"
                        "5. Provide an overall emotional assessment\n"
                        "What emotions are being expressed?"
                    )
                else:
                    base_prompt = (
                        "Describe the emotional content of this scene. "
                        "Focus on the visible emotions and their expressions."
                    )
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
            
            # 10. Response processing
            print("\n=== Response Processing ===")
            response = self.model["processor"].decode(
                output[0], 
                skip_special_tokens=True
            ).strip()
            print(f"Raw response: {response}")
            
            # 11. Parse response and remove prompt text
            # Remove prompt text if it appears in the response
            base_prompts = [
                "Analyze this video clip in detail:",
                "1. Describe the visible emotions and expressions",
                "2. Note any changes in emotional state",
                "3. Describe relevant body language and gestures",
                "4. Mention the setting and context if relevant",
                "5. Provide an overall emotional assessment",
                "What emotions are being expressed?",
                "Describe the emotional content of this scene.",
                "Focus on the visible emotions and their expressions.",
                "What is the emotion in this content?",
                "Use ONLY these labels:"
            ]
            
            cleaned_response = response
            for prompt in base_prompts:
                cleaned_response = cleaned_response.replace(prompt, "").strip()
            
            # Remove any leading/trailing punctuation and numbers
            cleaned_response = re.sub(r'^\d+\.\s*', '', cleaned_response)
            cleaned_response = cleaned_response.strip('.:,;!? ')
            
            # Split analysis into sections if available
            sections = re.split(r'\d+\.\s*', cleaned_response)
            sections = [s.strip() for s in sections if s.strip()]
            
            if len(sections) >= 3:  # Detailed analysis available
                emotion_details = {
                    'expressions': sections[0] if len(sections) > 0 else "",
                    'emotional_changes': sections[1] if len(sections) > 1 else "",
                    'body_language': sections[2] if len(sections) > 2 else "",
                    'context': sections[3] if len(sections) > 3 else "",
                    'overall_assessment': sections[4] if len(sections) > 4 else ""
                }
                
                # Extract primary emotion from overall assessment
                category = "DETAILED"
                explanation = cleaned_response
            else:
                if ":" in cleaned_response:
                    category, explanation = cleaned_response.split(":", 1)
                    category = category.strip().upper()
                    if category.startswith("[") and category.endswith("]"):
                        category = category[1:-1]
                    explanation = explanation.strip()
                else:
                    category = "OBSERVATION"
                    explanation = cleaned_response.strip()
                emotion_details = None
            
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
                "emotion_details": emotion_details,
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
    