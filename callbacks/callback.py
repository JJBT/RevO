import logging


logger = logging.getLogger(__name__)


class Callback:
    def __init__(self, frequency=0, before=False, after=False,
                 attributes=None):

        if frequency < 0:
            raise ValueError("Frequency argument should be positive.")

        if attributes is None:
            self._attributes = dict()
        else:
            self._attributes = attributes

        self.frequency = frequency
        self.before = before
        self.after = after

    def __call__(self, trainer):
        raise NotImplementedError()

    def before_run(self, trainer):
        if self.before:
            self.__call__(trainer)

    def after_run(self, trainer):
        if self.after:
            self.__call__(trainer)

    def set_trainer(self, trainer):
        for attribute_name in self._attributes:
            attribute_default_value = self._attributes[attribute_name]
            trainer.state.add_attribute(attribute_name, attribute_default_value)
