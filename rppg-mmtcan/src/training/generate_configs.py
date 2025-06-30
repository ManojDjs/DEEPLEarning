import os
import yaml

# Define your models and datasets
models = ["tscan", "mtts_can", "deep_phys", "physnet"]
datasets = ["UBFC", "PURE", "SNR", "BPD4"]

# Define a base template for YAML configs
def create_config(model, dataset):
    return {
        "model": {
            "name": model,
            "input_size": [3, 36, 36],  # common size
            "frames": 180,
            "dropout": 0.25 if model == "mtts_can" else 0.0,
            "num_classes": 1,
            "sampling_rate": 30,
        },
        "dataset": {
            "name": dataset,
            "video_path": f"./datasets/{dataset}/videos",
            "label_path": f"./datasets/{dataset}/gtdump.xmp",
            "format": "rgb",
            "clip_length": 180,
            "step": 30,
        },
        "training": {
            "epochs": 50,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss_fn": "mse",
            "shuffle": True,
        },
        "output": {
            "save_dir": f"./saved_models/{model}_{dataset}"
        }
    }

# Make the configs directory
os.makedirs("configs", exist_ok=True)

# Generate and write all YAML configs
for model in models:
    for dataset in datasets:
        config = create_config(model, dataset)
        filename = f"configs/{model}_{dataset}.yaml"
        with open(filename, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✅ Created: {filename}")
