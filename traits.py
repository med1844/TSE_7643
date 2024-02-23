from abc import abstractmethod
from typing import Dict, Any

Json = Dict[str, Any]


class SerdeJson:
    @abstractmethod
    def to_json(self) -> Json:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, obj: Json):
        pass
