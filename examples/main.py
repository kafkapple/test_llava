import cv2
import numpy as np
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv
import pandas as pd
import re
from tqdm import tqdm
from glob import glob

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.pipeline.emotion_pipeline import EmotionPipeline

# Hugging Face 토큰으로 로그인
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# 모델 저장 경로 설정
MODEL_PATH = Path("E:/models/LLaVA-NeXT-Video-7B-32K-hf")

def ensure_model():
    """모델 파일 확인 및 다운로드"""
    if not os.path.exists(MODEL_PATH):
        print(f"모델 다운로드 중... 경로: {MODEL_PATH}")
        snapshot_download(
            repo_id="llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
            local_dir=MODEL_PATH,
            token=HF_TOKEN,
            ignore_patterns=["*.md", "*.txt"]
        )
    else:
        print(f"기존 모델 사용: {MODEL_PATH}")
    return MODEL_PATH

def load_video(path):
    """비디오 파일을 프레임 배열로 로드"""
    cap = cv2.VideoCapture(path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 8프레임 균등 추출
    frame_indices = np.linspace(0, total_frames-1, 8, dtype=int)
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_indices:
            frames.append(frame)
    cap.release()
    return np.array(frames)

def load_image(path):
    """이미지 파일을 numpy 배열로 로드"""
    return cv2.imread(path)

def extract_emotion_and_confidence(text):
    """모델 출력에서 감정 단어와 신뢰도 추출"""
    # 감정 단어 추출 (첫 번째 단어)
    emotion = text.split()[0].strip('.:,;!?').lower()
    
    # 신뢰도 추정 (설명의 확실성에 기반)
    confidence_words = {
        'clearly': 0.9, 'definitely': 0.9, 'obviously': 0.9,
        'appears': 0.7, 'seems': 0.7, 'likely': 0.7,
        'might': 0.5, 'could': 0.5, 'possibly': 0.5,
        'unclear': 0.3, 'uncertain': 0.3, 'hard to tell': 0.3
    }
    
    confidence = 0.7  # 기본값
    for word, conf in confidence_words.items():
        if word in text.lower():
            confidence = conf
            break
            
    return emotion, confidence

def process_directory(directory_path, pipeline, file_types=('*.jpg', '*.jpeg', '*.png', '*.mp4', '*.mov')):
    """디렉토리 내의 모든 이미지/비디오 파일 처리"""
    # 디렉토리 존재 확인 및 생성
    directory_path = os.path.abspath(directory_path)
    if not os.path.exists(directory_path):
        print(f"디렉토리 생성: {directory_path}")
        os.makedirs(directory_path)
    
    results = []
    
    # 모든 파일 리스트 수집
    files = []
    for file_type in file_types:
        files.extend(glob(os.path.join(directory_path, '**', file_type), recursive=True))
    
    if not files:
        print(f"경고: {directory_path}에서 처리할 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    
    print(f"\n총 {len(files)}개 파일 처리 시작...")
    
    for file_path in tqdm(files):
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # 파일 타입에 따라 처리
            if file_ext in ['.jpg', '.jpeg', '.png']:
                data = {"image": load_image(file_path)}
            else:  # 비디오 파일
                data = {"video": load_video(file_path)}
            
            # 감정 분석 실행
            result = pipeline.process(data)
            
            if result["success"]:
                # 감정 단어와 신뢰도 추출
                emotion, confidence = extract_emotion_and_confidence(result["emotion"])
                
                results.append({
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'type': 'image' if file_ext in ['.jpg', '.jpeg', '.png'] else 'video',
                    'emotion': emotion,
                    'confidence': confidence,
                    'full_description': result["emotion"]
                })
            else:
                print(f"\n처리 실패 - {file_path}: {result['emotion']}")
                
        except Exception as e:
            print(f"\n에러 발생 - {file_path}: {str(e)}")
    
    if not results:
        print("경고: 처리된 결과가 없습니다.")
        return pd.DataFrame()
    
    # 결과 저장 디렉토리 생성
    output_dir = os.path.join(directory_path, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 타임스탬프를 파일명에 추가
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'emotion_analysis_results_{timestamp}.csv')
    
    # 데이터프레임 생성 및 저장
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n분석 결과가 저장되었습니다: {output_path}")
    return df

def main():
    # 모델 준비
    model_path = ensure_model()
    print(f"모델 경로: {model_path}")
    
    # 파이프라인 초기화
    pipeline = EmotionPipeline(model_path=model_path)
    
    while True:
        print("\n=== 감정 분석 모드 선택 ===")
        print("1. 단일 파일 처리")
        print("2. 디렉토리 일괄 처리")
        print("3. 종료")
        
        choice = input("\n선택하세요 (1-3): ").strip()
        
        if choice == "1":
            # 단일 파일 처리
            print("\n=== 단일 파일 감정 분석 ===")
            file_path = input("파일 경로를 입력하세요 (이미지 또는 비디오): ").strip()
            if not file_path:
                file_path = "E:/data/emotion/happy/happy_Image_2.jpg"  # 기본값
                
            file_ext = os.path.splitext(file_path)[1].lower()
            try:
                if file_ext in ['.jpg', '.jpeg', '.png']:
                    print("\n이미지 처리 중...")
                    data = {"image": load_image(file_path)}
                elif file_ext in ['.mp4', '.mov']:
                    print("\n비디오 처리 중...")
                    data = {"video": load_video(file_path)}
                else:
                    print("지원하지 않는 파일 형식입니다.")
                    continue
                
                result = pipeline.process(data)
                if result["success"]:
                    emotion, confidence = extract_emotion_and_confidence(result["emotion"])
                    print(f"\n파일: {os.path.basename(file_path)}")
                    print(f"감정: {emotion}")
                    print(f"신뢰도: {confidence:.2f}")
                    print(f"상세 설명: {result['emotion']}")
                else:
                    print(f"처리 실패: {result['emotion']}")
                    
            except Exception as e:
                print(f"에러 발생: {str(e)}")
            
        elif choice == "2":
            # 디렉토리 일괄 처리
            print("\n=== 디렉토리 일괄 처리 ===")
            default_path= "E:/data/mer2024/test"
            target_dir = input(f"처리할 디렉토리 경로를 입력하세요: default path: {default_path} ").strip()
            if not target_dir:
                target_dir =  default_path # 기본값
            
            results_df = process_directory(target_dir, pipeline)
            
            if not results_df.empty:
                print("\n=== 분석 결과 요약 ===")
                print("\n감정 분포:")
                print(results_df['emotion'].value_counts())
                print(f"\n평균 신뢰도: {results_df['confidence'].mean():.2f}")
                
                print("\n신뢰도 상위 5개 항목:")
                top_5 = results_df.nlargest(5, 'confidence')
                for _, row in top_5.iterrows():
                    print(f"\n파일: {row['file_name']}")
                    print(f"감정: {row['emotion']} (신뢰도: {row['confidence']:.2f})")
                    print(f"설명: {row['full_description']}")
            else:
                print("\n처리된 결과가 없습니다.")
                
        elif choice == "3":
            print("\n프로그램을 종료합니다.")
            break
            
        else:
            print("\n잘못된 선택입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main() 