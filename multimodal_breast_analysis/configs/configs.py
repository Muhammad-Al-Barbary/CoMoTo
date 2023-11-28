import json
import wandb
import datetime


with open(
    "multimodal_breast_analysis/configs/configs.json"
) as json_path:
    config_dict = json.load(json_path)
    wandb.init(
        project='multimodal_breast_analysis', 
        config=config_dict,
        name=datetime.datetime.now().strftime("%H%M%S_%d%m%Y")
        )
config = wandb.config