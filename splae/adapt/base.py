from abc import ABC, abstractmethod
import torch

from segment_anything import sam_model_registry

class PLGeneratorBase(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs):
        """Must be implemented in subclasses to provide the generation logic."""
        pass

class SamPLGeneratorBase(ABC):
    def __init__(self,
                 model_type,
                 weights_path,
                 image_size,
                 device,
                 encoder_adapter=True,
                 visualizer=None):
        self.model_type = model_type
        self.weights_path = weights_path
        self.image_size = image_size
        self.device = device
        self.encoder_adapter = encoder_adapter
        self.model = sam_model_registry.get(model_type)(image_size, weights_path, encoder_adapter).to(device)
        self.visualizer = visualizer

    @property
    @abstractmethod
    def predictor(self):
        """Must be implemented in subclasses to provide access to the predictor."""
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Must be implemented in subclasses to provide the generation logic."""
        pass