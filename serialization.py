

"""Interfaces common to all Neural Modules and Models."""
import traceback
from abc import ABC
from typing import  Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import copy
from omegaconf import errors as omegaconf_errors
_HAS_HYDRA = True

__all__ = [ 'Serialization']

def _convert_config(cfg: 'OmegaConf'):
    """ Recursive function convertint the configuration from old hydra format to the new one. """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        exit(1)

    # Get rid of cls -> _target_.
    if 'cls' in cfg and '_target_' not in cfg:
        cfg._target_ = cfg.pop('cls')

    # Get rid of params.
    if 'params' in cfg:
        params = cfg.pop('params')
        for param_key, param_val in params.items():
            cfg[param_key] = param_val

    # Recursion.
    try:
        for _, sub_cfg in cfg.items():
            if isinstance(sub_cfg, DictConfig):
                _convert_config(sub_cfg)
    except omegaconf_errors.OmegaConfBaseException as e:
        logging.warning(f"Skipped conversion for config/subconfig:\n{cfg}\n Reason: {e}.")


def maybe_update_config_version(cfg: 'DictConfig'):
    """
    Recursively convert Hydra 0.x configs to Hydra 1.x configs.

    Changes include:
    -   `cls` -> `_target_`.
    -   `params` -> drop params and shift all arguments to parent.
    -   `target` -> `_target_` cannot be performed due to ModelPT injecting `target` inside class.

    Args:
        cfg: Any Hydra compatible DictConfig

    Returns:
        An updated DictConfig that conforms to Hydra 1.x format.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        exit(1)
    if cfg is not None and not isinstance(cfg, DictConfig):
        try:
            temp_cfg = OmegaConf.create(cfg)
            cfg = temp_cfg
        except omegaconf_errors.OmegaConfBaseException:
            # Cannot be cast to DictConfig, skip updating.
            return cfg

    # Make a copy of model config.
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)

    # Convert config.
    _convert_config(cfg)

    # Update model config.
    OmegaConf.set_struct(cfg, True)

    return cfg


def import_class_by_path(path: str):
    """
    Recursive import of class by path string.
    """
    paths = path.split('.')
    path = ".".join(paths[:-1])
    class_name = paths[-1]
    mod = __import__(path, fromlist=[class_name])
    mod = getattr(mod, class_name)
    return mod



class Serialization(ABC):
    @classmethod
    def from_config_dict(cls, config: 'DictConfig', trainer: Optional['Trainer'] = None):
        """Instantiates object using DictConfig-based configuration"""
        # Resolve the config dict
        if _HAS_HYDRA:
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
                config = OmegaConf.create(config)
                OmegaConf.set_struct(config, True)

            config = maybe_update_config_version(config)

        # Hydra 0.x API
        if ('cls' in config or 'target' in config) and 'params' in config and _HAS_HYDRA:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        # Hydra 1.x API
        elif '_target_' in config and _HAS_HYDRA:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        else:
            instance = None
            imported_cls_tb = None
            instance_init_error = None
            # Attempt class path resolution from config `target` class (if it exists)
            if 'target' in config:
                target_cls = config["target"]  # No guarantee that this is a omegaconf class
                imported_cls = None
                try:
                    # try to import the target class
                    imported_cls = import_class_by_path(target_cls)
                except Exception:
                    imported_cls_tb = traceback.format_exc()

                # try instantiating model with target class
                if imported_cls is not None:
                    # if calling class (cls) is subclass of imported class,
                    # use subclass instead
                    if issubclass(cls, imported_cls):
                        imported_cls = cls

                    try:
                        instance = imported_cls(cfg=config)

                    except Exception as e:
                        imported_cls_tb = traceback.format_exc()
                        instance_init_error = str(e)
                        instance = None

            # target class resolution was unsuccessful, fall back to current `cls`
            if instance is None:
                if imported_cls_tb is not None:
                    logging.debug(
                        f"Model instantiation from target class {target_cls} failed with following error.\n"
                        f"Falling back to `cls`.\n"
                        f"{imported_cls_tb}"
                    )
                
                instance = cls(cfg=config, trainer=trainer)
                try:
                    instance = cls(cfg=config)
                except Exception as e:
                    if imported_cls_tb is not None:
                        logging.error(f"Instance failed restore_from due to: {instance_init_error}")
                        logging.error(f"{imported_cls_tb}")
                    raise e

        if not hasattr(instance, '_cfg'):
            instance._cfg = config
        return instance

   