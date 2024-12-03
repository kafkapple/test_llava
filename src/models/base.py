from abc import ABC, abstractmethod
from typing import Dict, Any

class MultiModalModel(ABC):
    """기본 멀티모달 모델 클래스"""
    
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> str:
        """감정 예측"""
        pass
    
    @abstractmethod
    def get_embedding(self, inputs: Dict[str, Any]) -> Any:
        """임베딩 추출"""
        pass 