from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class EmotionEntry:
    """감정 데이터 엔트리"""
    id: str
    emotion: str
    embeddings: Dict[str, np.ndarray]
    metadata: Dict[str, Any]

class SimpleVectorStore:
    """간단한 인메모리 벡터 저장소"""
    
    def __init__(self):
        self.entries: List[EmotionEntry] = []
    
    def add_entry(self, entry: EmotionEntry):
        """새로운 엔트리 추가"""
        self.entries.append(entry)
    
    def search_similar(
        self, 
        query_embedding: np.ndarray,
        modality: str,
        top_k: int = 5
    ) -> List[EmotionEntry]:
        """코사인 유사도 기반 검색"""
        similarities = []
        for entry in self.entries:
            if modality in entry.embeddings:
                similarity = np.dot(
                    query_embedding, 
                    entry.embeddings[modality]
                ) / (
                    np.linalg.norm(query_embedding) * 
                    np.linalg.norm(entry.embeddings[modality])
                )
                similarities.append((similarity, entry))
        
        # 상위 k개 반환
        return [
            entry for _, entry in sorted(
                similarities, 
                key=lambda x: x[0], 
                reverse=True
            )[:top_k]
        ] 