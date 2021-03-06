from callbacks.callback import Callback


class ValidationCallback(Callback):
    def __init__(self, metrics, frequency, dataloader=None):
        self._metrics = {f'last_validation_{metric.name}': metric.default_value for metric in metrics}
        super().__init__(frequency=frequency, before=False, after=True, attributes=self._metrics)

        self.dataloader = dataloader
        self.metrics = metrics

    def __call__(self, trainer):
        self._log_call(trainer)
        self.computed_metrics = trainer.evaluate(dataloader=self.dataloader, metrics=self.metrics)
        for metric_name, metric_value in self.computed_metrics.items():
            setattr(trainer.state, f'last_validation_{metric_name}', metric_value)
