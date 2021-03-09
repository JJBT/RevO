from callbacks.callback import Callback


class LogCallback(Callback):
    def __init__(self, frequency):
        super().__init__(frequency=frequency, before=False, after=True)

    def __call__(self, trainer):
        trainer.state.log_info()
