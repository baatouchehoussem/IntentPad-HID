from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class AdapterFrame:
    left_stick: Tuple[float, float] = (0.0, 0.0)
    right_stick: Tuple[float, float] = (0.0, 0.0)
    triggers: Tuple[float, float] = (0.0, 0.0)
    buttons: Dict[str, bool] = None
    annotated_frame: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.buttons is None:
            self.buttons = {}
        if self.metadata is None:
            self.metadata = {}


class InputAdapter(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._running = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def read_frame(self) -> AdapterFrame:
        pass

    def is_running(self) -> bool:
        return self._running

    def update_config(self, config: Dict[str, Any]) -> None:
        self._config.update(config)