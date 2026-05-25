import json
import yaml


class Args(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return Args(value)
        else:
            return value

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = Args(value)
        self[key] = value

    def merge(self, other):
        if isinstance(other, Args):
            for key, value in other.items():
                self.__setattr__(key, value)
        elif isinstance(other, dict):
            for key, value in other.items():
                self.__setattr__(key, value)
        else:
            raise ValueError("Invalid update source. Expected Args or dict.")

    
    def __str__(self):
        return json.dumps(self, indent=5)
    
    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(data)
    
    
if __name__ == "__main__":
    args = Args.from_yaml("/home/light_sun/workspace/inrsteg/FastTools/framework/_cfg.yaml")
    print(args.b)

    pass