import json
import wandb
import datetime

def load_configs(config_name = None):
    with open(
        "multimodal_breast_analysis/configs/configs.json" if config_name is None else "multimodal_breast_analysis/configs/" + config_name + ".json"
    ) as json_path:
        config_dict = json.load(json_path)
        wandb.init(
            project='multimodal_breast_analysis', 
            config=config_dict,
            name=datetime.datetime.now().strftime("%H%M%S_%d%m%Y") if config_name is None else config_name
            )
    config = wandb.config
    return config