from copy import deepcopy
import copy
from typing import Any, Dict, Optional
import yaml
from omegaconf import Container, OmegaConf, DictConfig

class Args(Dict):
    """Extended dictionary accessible with dot notation.

    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(new_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "my-key":  3.14
    "new_key": 42

    """

    def __getattr__(self, key: str) -> Optional[Any]:
        return self.get(key, None)
        try:
            return self[key]
        except KeyError as exp:
            return None
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __repr__(self) -> str:
        return super().__repr__()

    # 添加 __deepcopy__ 方法
    def __deepcopy__(self, memo):
        new_copy = Args()
        for key, value in self.items():
            new_copy[key] = copy.deepcopy(value, memo)
        return new_copy
      
    def merge(self, other: 'Args') -> 'Args':
        """Merge two Args.

        >>> ad1 = Args({'key1': 1, 'key2': 'abc'})
        >>> ad2 = Args({'key2': 'def', 'key3': 3.14})
        >>> ad1.merge(ad2)
        "key1":    1
        "key2":    def
        "key3":    3.14
        """
        for key, value in other.items():
            if key in self and isinstance(value, Dict):
                self[key].merge(value)
            else:
                self[key] = value
        return self
    
    def load(self, yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_params = yaml.safe_load(f)
            if yaml_params is not None:
                self.load_from_dict(yaml_params)
            pass
        return self
    
    def load_from_dict(self, n_params: dict):
        self._load_dict(self, n_params)
        pass

    def _load_dict(self, params, dict_params: dict):
        for key, item in dict_params.items():
            if isinstance(item, dict):
                params[key] = Args()
                self._load_dict(params[key], item)
                pass
            else:
                params[key] = item
                pass
            pass
        pass




# class Args:
#     def __init__(self):
#         self.config = OmegaConf.create({})

#     def load(self, path):
#         self.config = OmegaConf.load(path)
#         OmegaConf.set_struct(self.config, False)  # 允许访问不存在的键
#         pass

#     def __getattr__(self, item):
#         # 递归访问配置中的键
#         def recursive_get(config, key):
#             if isinstance(config, DictConfig) or key in config:
#                 return config[key]
#             else:
#                 return None

#         # 顶层访问
#         return recursive_get(self.config, item)

#     def __setattr__(self, key, value):
#         # 如果 key 是 config，使用父类的 __setattr__
#         if key == "config":
#             super().__setattr__(key, value)
#         else:
#             # 否则将属性设置到 config 中
#             self.config[key] = value

#     def __repr__(self) -> str:
#         return self.config.__repr__()


if __name__ == "__main__":
    args = Args({"a": 10})
    args.load("/home/light_sun/workspace/inrmark/cfg/main_v2.yaml")
    b = deepcopy(args)
