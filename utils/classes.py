from utils.fonctions import load_config


class Config:
    def __init__(self, config_dict):
        if not isinstance(config_dict, dict):
            raise ValueError('Config must be initialized with a dictionary...')

        self.config = config_dict
        self._recursive_objectify(self.config)

    def _recursive_objectify(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = self._make_subconfig(value)

            setattr(self, key, value)

    @staticmethod
    def _make_subconfig(dictionary):
        subconfig = type('SubConfig', (), {})

        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = Config._make_subconfig(v)

            setattr(subconfig, k, v)

        return subconfig

    @staticmethod
    def merge(config1, config2):
        def merge_dicts(d1, d2):
            for key in d2:
                if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    merge_dicts(d1[key], d2[key])
                else:
                    d1[key] = d2[key]

        new_dict = dict(config1.config)
        merge_dicts(new_dict, config2.config)

        return Config(new_dict)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.config})'


class RunDemo:
    def __init__(self, config_file: str, id: str, name: str, sub_config: str = None) -> None:
        self.config = load_config(config_file, sub_config)
        self.name = name
        self.id = id
