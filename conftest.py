"""This file is principlay a relay between the main files and tests directory"""

"""for now this is a simple config test but i should includ it in the test ditectory """

"""Think of creating a main test file in the root that calls the different tests in the test directory"""

import pytest
import hydra
import logging
from omegaconf.omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from tests.test_configs import test_hydra_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="./config", config_name="config",version_base=None)
def test_hydra_config(cfg):
    """Test Hydra configuration using the @hydra.main decorator"""
    # Convert to Python native types
    cfg_dict = OmegaConf.to_yaml(cfg, resolve=True)
    
    # Check main sections exist
    assert "app" in cfg_dict
    assert "model" in cfg_dict
    assert "web_search" in cfg_dict
    assert "processing" in cfg_dict
    
    # Check model config
    assert isinstance(cfg_dict["model"]["available_models"], list)
    assert len(cfg_dict["model"]["available_models"]) > 0
    assert cfg_dict["model"]["default_model"] in cfg_dict["model"]["available_models"]
    
    # Check web search config
    assert isinstance(cfg_dict["web_search"]["use_web_search"], bool)
    return True  # Return value needed for hydra.main

def test_config_loading():
    """Wrapper test that runs the Hydra test function"""
    # Run the Hydra test and verify it completes successfully
    assert test_hydra_config()