import os
import torch
from pathlib import Path
import importlib
from ..utils import remote_data_loader
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs"

def train_model_from_config(config_path):
    print(f"🔧 Loading config: {config_path}")
    config = CONFIG_PATH

    # Step 1: Dynamically import the data loader based on the model and dataset specified in the config
    model_name = config['MODEL']['NAME'].lower()
    dataset_name = config['DATASET']['NAME'].lower()

    dataloader_module = importlib.import_module(f'src.utils.dataloader.{model_name}_{dataset_name}')
    dataloader_class = getattr(dataloader_module, f'{model_name.capitalize()}Loader')  # Assuming class name follows this pattern

    # Instantiate the data loader
    data_loader = dataloader_class(name=model_name, data_path=config['DATASET']['PATH'], config_data=config)

    # Build train and validation dataloaders
    train_loader, val_loader = data_loader.get_dataloaders()

    # Step 2: Train the model
    model, best_metric = data_loader.train(config, train_loader, val_loader)

    # Step 3: Save the model
    save_path = config['MODEL']['SAVE_PATH']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"✅ Model {config['MODEL']['NAME']} saved to {save_path} | Best Metric: {best_metric:.4f}\n")


def main():
    models = ["tscan", "mttscan", "deepphys", "physnet"]
    datasets = ["pure", "ubfc", "snr", "bpd4"]

    config_paths = [
        f"configs/train_{dataset}_{model}.yaml"
        for dataset in datasets
        for model in models
    ]

    for config_path in config_paths:
        train_model_from_config(config_path)


if __name__ == "__main__":
    main()
