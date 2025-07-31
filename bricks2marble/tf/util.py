from typing import TypeVar

import tensorflow as tf

T_Model = TypeVar("T_Model")


def is_hooked(model_cls: type[T_Model]) -> type[T_Model]:

    def hooks(self) -> dict:
        if not hasattr(self, "_hooks"):
            self._hooks = {}
            self._hooks_active = False
        return self._hooks
    setattr(model_cls, "hooks", property(hooks))

    def hook(self, name: str, x: tf.Tensor) -> tf.Tensor:
        if self._hooks_active:
            self.hooks.update({name: tf.identity(x)})
        return x
    setattr(model_cls, "hook", hook)

    def attach_hooks(self) -> None:
        self._hooks_active = True
    setattr(model_cls, "attach_hooks", attach_hooks)

    def release_hooks(self) -> None:
        self._hooks_active = False
    setattr(model_cls, "release_hooks", release_hooks)

    def clear_hooks(self) -> None:
        self.hooks = {}
    setattr(model_cls, "clear_hooks", clear_hooks)

    return model_cls
