from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    A data class to hold configuration parameters for data ingestion.

    Attributes:
        root_dir (Path): The root directory for the project.
        source_URL (str): The URL from which to download the data.
        local_data_file (Path): The file path to save the downloaded data locally.
        unzip_dir (Path): The directory path to unzip the downloaded data.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

    




@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """
    A data class to hold configuration parameters for preparing and updating a base model.

    Attributes:
        root_dir (Path): The root directory for the project.
        base_model_path (Path): The file path to save the base model.
        updated_base_mode_path (Path): The file path to save the updated base model.
        params_image_size (list): The size of input images (e.g., [height, width, channels]).
        params_learning_rate (float): The learning rate for the optimizer.
        params_include_top (bool): Whether to include the fully connected layer at the top of the network.
        params_weights (str): The pre-trained weights to initialize the model.
        params_classes (int): The number of output classes for classification.
    """
    root_dir: Path
    base_model_path: Path
    updated_base_mode_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir : Path
    trained_model_path : Path
    updated_base_model_path : Path
    training_data: Path
    params_epochs : int
    params_batch_size : int
    params_is_augmentation : bool
    params_image_size : int
    
@dataclass(frozen =True)
class EvaluationConfig:
    path_of_model : Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    