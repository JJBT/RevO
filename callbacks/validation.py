from callbacks.callback import Callback


class ValidationCallback(Callback):
    def __init__(self, frequency):
        super().__init__(frequency=frequency, before=False, after=True)

    def __call__(self, trainer):
        val_dataloader = trainer.val_dataloader
        for name in val_dataloader:
            dataloader = val_dataloader[name]['dataloader']
            self.computed_metrics = trainer.evaluate(dataloader=dataloader, metrics=trainer.metrics)
            for metric_name, metric_value in self.computed_metrics.items():
                trainer.state.add_validation_metric(name=f'{name}/{metric_name}', value=metric_value)

        trainer.state.log_validation()

        if 'TensorBoardCallback' in trainer.callbacks:
            tb_callback = trainer.callbacks['TensorBoardCallback']
            tb_callback.add_validation_metrics(trainer)
            tb_callback.draw_prediction(trainer)
