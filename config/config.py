from dataclasses import dataclass
from typing import List
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    available_models: List[str]
    default_model: str
    temperature: float

@dataclass
class AppConfig:
    collection_name: str
    rag_enabled: bool
    similarity_threshold: float

@dataclass
class WebSearchConfig:
    use_web_search: bool
    exa_api_key: str
    custom_domains: List[str]

@dataclass
class Config:
    app: AppConfig
    model: ModelConfig
    web_search: WebSearchConfig

def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)

def convert_to_config(cfg_dict: dict) -> Config:
    """Convert dictionary to Config dataclass"""
    return Config(
        app=AppConfig(**cfg_dict['app']),
        model=ModelConfig(**cfg_dict['model']),
        web_search=WebSearchConfig(**cfg_dict['web_search'])
    )