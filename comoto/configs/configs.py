import json
import wandb
import datetime

def load_configs(config_name = None, wandb_log = True):
    """
    Reads the configuration file and creates a configuration object
    Args:
        config_name: String: the configuration file name to be loaded. Default reads the file names "configs"
        wandb_log: Whether to instantiate WANDB or not. Default is True.
    """
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
    """
    class for reading configuration from the json file if wandb is off
    """
    def __init__(self, config_dict) -> None:
        self.wandb = False
        for key in config_dict:
            setattr(self, key, config_dict[key])