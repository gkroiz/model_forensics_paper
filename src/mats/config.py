"""
Config loading utilities for experiment runners.

Provides OmegaConf-based config loading with:
- Custom YAML representer for multiline strings
- Custom resolvers (e.g., pluralize)
- Config loading with CLI overrides
- Config resolution for containers
"""

from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf


# =============================================================================
# YAML SETUP
# =============================================================================

def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
    """Custom YAML representer for multiline strings (use literal block style)."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_representer)


# =============================================================================
# OMEGACONF RESOLVERS — GENERAL
# =============================================================================

OmegaConf.register_new_resolver(
    "pluralize",
    lambda num, singular, plural: f"{num} {singular if num == 1 else plural}",
    replace=True,
)

# =============================================================================
# OMEGACONF RESOLVERS — SECRET NUMBER
# =============================================================================

OmegaConf.register_new_resolver(
    "pct",
    lambda x: int(float(x) * 100),
    replace=True,
)

OmegaConf.register_new_resolver(
    "pct_complement",
    lambda x: int((1 - float(x)) * 100),
    replace=True,
)


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """
    Load config from file and apply CLI overrides.
    
    Args:
        config_path: Path to the YAML config file
        overrides: List of dotlist overrides (e.g., ["agent.model=foo", "task.variation=bar"])
    
    Returns:
        Merged OmegaConf DictConfig
    """
    cfg = OmegaConf.load(config_path)
    
    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_conf)
    
    return cfg


def resolve_config(cfg: DictConfig) -> str:
    """
    Resolve config interpolations and return as YAML string.
    
    Restructures the config to match the expected format for containers:
    environment, agent, task, prompts
    
    Args:
        cfg: OmegaConf DictConfig (possibly with unresolved interpolations)
    
    Returns:
        Resolved config as YAML string
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    
    output = {
        "environment": resolved.get("environment"),
        "agent": resolved.get("agent", {}),
        "task": resolved.get("task", {}),
        "prompts": resolved.get("prompts", {}),
    }
    
    return yaml.dump(output, default_flow_style=False, sort_keys=False)


def write_resolved_config(cfg: DictConfig, path: Path) -> None:
    """
    Write the fully resolved config to a YAML file.
    
    Args:
        cfg: OmegaConf DictConfig to resolve and write
        path: Path to write the resolved config
    """
    resolved_yaml = resolve_config(cfg)
    path.write_text(resolved_yaml)
