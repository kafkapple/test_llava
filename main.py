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
from src.models.model_factory import ModelManager

# Load environment variables
load_dotenv()

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.pipeline.emotion_pipeline import EmotionPipeline

# Login with Hugging Face token
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")


def load_video(path):
    """Load video file as frame array"""
    cap = cv2.VideoCapture(path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract 8 frames evenly
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
    """Load image file as numpy array"""
    return cv2.imread(path)


def extract_emotion_and_confidence(text):
    """Extract emotion word and confidence from model output"""
    # Extract emotion word (first word)
    emotion = text.split()[0].strip('.:,;!?').lower()
    
    # Estimate confidence (based on certainty in explanation)
    confidence_words = {
        'clearly': 0.9, 'definitely': 0.9, 'obviously': 0.9,
        'appears': 0.7, 'seems': 0.7, 'likely': 0.7,
        'might': 0.5, 'could': 0.5, 'possibly': 0.5,
        'unclear': 0.3, 'uncertain': 0.3, 'hard to tell': 0.3
    }
    
    confidence = 0.7  # default value
    for word, conf in confidence_words.items():
        if word in text.lower():
            confidence = conf
            break
            
    return emotion, confidence


def process_directory(directory_path, pipeline, file_types=('*.jpg', '*.jpeg', '*.png', '*.mp4', '*.mov')):
    """Process all image/video files in directory"""
    # Check and create directory
    directory_path = os.path.abspath(directory_path)
    if not os.path.exists(directory_path):
        print(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)
    
    results = []
    
    # Collect all files
    files = []
    for file_type in file_types:
        files.extend(glob(os.path.join(directory_path, '**', file_type), recursive=True))
    
    if not files:
        print(f"Warning: No files found to process in {directory_path}")
        return pd.DataFrame()
    
    print(f"\nStarting processing of {len(files)} files...")
    
    for file_path in tqdm(files):
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Process based on file type
            if file_ext in ['.jpg', '.jpeg', '.png']:
                data = {"image": load_image(file_path)}
            else:  # video file
                data = {"video": load_video(file_path)}
            
            # Run emotion analysis
            result = pipeline.process(data)
            
            if result["success"]:
                # Extract emotion word and confidence
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
                print(f"\nProcessing failed - {file_path}: {result['emotion']}")
                
        except Exception as e:
            print(f"\nError occurred - {file_path}: {str(e)}")
    
    if not results:
        print("Warning: No results processed.")
        return pd.DataFrame()
    
    # Create results directory
    output_dir = os.path.join(directory_path, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'emotion_analysis_results_{timestamp}.csv')
    
    # Create and save DataFrame
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nAnalysis results saved to: {output_path}")
    return df


def main():
    # Prepare model
    # Set model path
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
    MODEL_PATH = "/home/joon/models"
    
    model_manager = ModelManager(token=HF_TOKEN)

    model_path = model_manager.ensure_model(
        model_id=model_id,
        save_directory=MODEL_PATH
    )
    
    # Initialize pipeline
    pipeline = EmotionPipeline(model_path=model_path)
    debug = True
    
    while True:
        print("\n=== Select Emotion Analysis Mode ===")
        print("1. Process Single File")
        print("2. Process Directory")
        print("3. Exit")
        if debug:
            print("4. Debug Mode")
            choice = "1"
        else:
            choice = input("\nSelect (1-3): ").strip()
        
        if choice == "1":
            # Process single file
            print("\n=== Single File Emotion Analysis ===")
            if debug:
                file_path = "/home/joon/Downloads/happy.jpeg" #""/home/joon/Downloads/angry.mp4" #"
            else:
                file_path = input("Enter file path (image or video): ").strip()
                if not file_path:
                    file_path = "/home/joon/Downloads/happy.jpeg"  # default
                    
            file_ext = os.path.splitext(file_path)[1].lower()
            try:
                if file_ext in ['.jpg', '.jpeg', '.png']:
                    print("\nProcessing image...")
                    data = {"image": load_image(file_path)}
                elif file_ext in ['.mp4', '.mov']:
                    print("\nProcessing video...")
                    data = {"video": load_video(file_path)}
                else:
                    print("Unsupported file format.")
                    continue
                
                result = pipeline.process(data)
                if result["success"]:
                    print(f"\nFile: {os.path.basename(file_path)}")
                    if result.get("emotion_details"):
                        print("\n=== Detailed Emotion Analysis ===")
                        details = result["emotion_details"]
                        print("\n1. Expressions and Emotions:")
                        print(f"   {details['expressions']}")
                        print("\n2. Emotional Changes:")
                        print(f"   {details['emotional_changes']}")
                        print("\n3. Body Language:")
                        print(f"   {details['body_language']}")
                        print("\n4. Context and Setting:")
                        print(f"   {details['context']}")
                        print("\n5. Overall Assessment:")
                        print(f"   {details['overall_assessment']}")
                    elif result["is_emotion"]:
                        print(f"Emotion: {result['emotion']}")
                        print(f"Explanation: {result['explanation']}")
                    else:
                        print("Emotion classification not possible")
                        print(f"Observation: {result['explanation']}")
                else:
                    print(f"Processing failed: {result['emotion']}")
                    
            except Exception as e:
                print(f"Error occurred: {str(e)}")
            if debug:
                break
            
        elif choice == "2":
            # Process directory
            print("\n=== Directory Processing ===")
            default_path = "home/joon/models"
            target_dir = input(f"Enter directory path to process (default: {default_path}): ").strip()
            if not target_dir:
                target_dir = default_path  # default value
            
            results_df = process_directory(target_dir, pipeline)
            
            if not results_df.empty:
                print("\n=== Analysis Summary ===")
                print("\nEmotion Distribution:")
                print(results_df['emotion'].value_counts())
                print(f"\nAverage Confidence: {results_df['confidence'].mean():.2f}")
                
                print("\nTop 5 Confidence Items:")
                top_5 = results_df.nlargest(5, 'confidence')
                for _, row in top_5.iterrows():
                    print(f"\nFile: {row['file_name']}")
                    print(f"Emotion: {row['emotion']} (Confidence: {row['confidence']:.2f})")
                    print(f"Description: {row['full_description']}")
            else:
                print("\nNo results processed.")
                
        elif choice == "3":
            print("\nExiting program.")
            break
            
        else:
            print("\nInvalid selection. Please try again.")


if __name__ == "__main__":
    main() 