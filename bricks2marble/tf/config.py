from typing import Any, Callable, TypeVar

import tensorflow as tf
from pydantic import BaseModel

T_Model = TypeVar("T_Model")


class ModelConfig(BaseModel):

    ...


def with_config(
    config_class: type[ModelConfig],
) -> Callable[[type[T_Model]], type[T_Model]]:

    def decorator(model_cls: type[T_Model]) -> type[T_Model]:

        def __init__(self, **kwargs) -> None:
            self.config = config_class(**kwargs)
            self._hooks_active = False
            self.hooks = {}

            if hasattr(self, "post_config_init"):
                self.post_config_init()
            else:
                super(model_cls, self).__init__()  # type: ignore

        def hook(self, name: str, x: tf.Tensor) -> tf.Tensor:
            if self._hooks_active:
                self.hooks.update({name: tf.identity(x)})
            return x

        def attach_hooks(self) -> None:
            self._hooks_active = True

        def release_hooks(self) -> None:
            self._hooks_active = False

        def clear_hooks(self) -> None:
            self.hooks = {}

        def get_config(self) -> dict[str, Any]:
            config_data = {
                field: getattr(self.config, field)
                for field in config_class.model_fields.keys()
            }
            try:
                config = super(model_cls, self).get_config()  # type: ignore
            except NotImplementedError:
                config = {}
            config.update(config_data)
            return config

        @classmethod
        def from_config(
            cls: type[T_Model],
            config: dict[str, Any],
            custom_objects: None = None,
        ) -> T_Model:
            model = cls(**config)
            if "input_shape" in config and hasattr(model, "build"):
                model.build(tuple(config["input_shape"]))  # type: ignore
            return model

        model_cls.__doc__ = (
            (model_cls.__doc__ or "") + (config_class.__doc__ or "")
        )
        model_cls.__init__ = __init__
        setattr(model_cls, "hook", hook)
        setattr(model_cls, "attach_hooks", attach_hooks)
        setattr(model_cls, "release_hooks", release_hooks)
        setattr(model_cls, "clear_hooks", clear_hooks)
        model_cls.get_config = get_config  # type: ignore
        model_cls.from_config = from_config
        return model_cls

    return decorator
