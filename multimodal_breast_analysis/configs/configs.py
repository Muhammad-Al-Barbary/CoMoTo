import json
import wandb
import datetime

def load_configs(config_name = None, wandb_log = True):
    with open(
        "multimodal_breast_analysis/configs/configs.json" if config_name is None else "multimodal_breast_analysis/configs/" + config_name + ".json"
    ) as json_path:
        config_dict = json.load(json_path)
        if wandb_log:
            wandb.init(
                project='multimodal_breast_analysis', 
                config=config_dict,
                name=datetime.datetime.now().strftime("%H%M%S_%d%m%Y") if config_name is None else config_name
                )
            config = wandb.config 
            config.wandb = True
        else:
            config = Config(config_dict)
    return config

class Config:
    """class for reading configuration from the json file."""
    def __init__(self, config_dict) -> None:
        self.wandb = False
        for key in config_dict:
            setattr(self, key, config_dict[key])