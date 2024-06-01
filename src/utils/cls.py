class Config:
    def __init__(self, *config_dicts):
        dictionary = {}
        for config in config_dicts:
            dictionary.update(config)

        self._recursive_objectify(dictionary)

    def _recursive_objectify(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)

            setattr(self, str(key), value)

    def to_dict(self):
        return {int(k): v for k, v in self.__dict__.items()}

    def __repr__(self):
        return f'{self.__class__.__name__}({self.config})'


class RunDemo:
    def __init__(self, config: Config, id: str, name: str) -> None:
        self.config = config
        self.name = name
        self.id = id


class TrainingMode:
    SEMI_SUPERVISED = 'semi_supervised'
    SUPERVISED = 'supervised'

class ModelName:
    MASK2FORMER = 'mask2former'
    SEGFORMER = 'segformer'