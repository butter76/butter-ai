from dataclasses import dataclass, asdict, field
import argparse
import yaml
from typing import TypeVar

T = TypeVar('T', bound='BaseConfig')

@dataclass
class BaseConfig:
    """Base configuration class that supports YAML and command line arguments."""
    def __post_init__(self):
        """Called after dataclass initialization to perform any additional setup."""
        pass

    @classmethod
    def from_yaml(cls: type[T], yaml_path: str) -> T:
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}  # Handle empty YAML files
        return cls(**config_dict)
    
    def save_yaml(self, yaml_path: str) -> None:
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, help='Path to YAML config file')
        for field in cls.__dataclass_fields__.values():
            parser.add_argument(
                f"--{field.name}",
                type=field.type,
                default=None,  # Changed to None to detect if arg was provided
                help=f"{field.name} (default: {field.default})"
            )
        return parser
    
    @classmethod
    def from_args_and_yaml(cls: type[T], configStr: str | None) -> T:
        parser = cls.get_parser()
        args = parser.parse_args()

        config_loc = configStr or args.config
        
        # Start with default config
        config_dict = {
            field.name: field.default 
            for field in cls.__dataclass_fields__.values()
        }
        
        # Layer in YAML config if provided
        if config_loc:
            with open(config_loc, 'r') as f:
                yaml_dict = yaml.safe_load(f) or {}  # Handle None case
                config_dict.update(yaml_dict)
        
        # Layer in command line arguments if provided
        cmd_dict = {
            k: v for k, v in vars(args).items()
            if k != 'config' and v is not None
        }
        config_dict.update(cmd_dict)
        
        return cls(**config_dict)
