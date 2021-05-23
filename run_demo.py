from demo.components import ImageInput, ImageOutput
from demo.interface import Interface
from demo.image_utils import preprocess_input, Patch

import hydra
from omegaconf import DictConfig, OmegaConf
import os
from settings import BASE_DIR
from callbacks import LoadCheckpointCallback
import signal
from utils.vis_utils import yet_another_draw_sample as draw


def create_model(cfg):
    from trainer import Trainer
    trainer = Trainer(cfg)
    ckpt_dir = os.path.join(BASE_DIR, cfg.path, 'checkpoints')
    trainer.register_callback(LoadCheckpointCallback(
        directory=ckpt_dir,
        filename=cfg.ckpt
    ))
    trainer._before_run_callbacks()
    return trainer.model


@hydra.main(config_path=os.path.join(BASE_DIR, 'conf'), config_name='config_valid')
def run(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    trainer_cfg_filename = os.path.join(BASE_DIR, cfg.path, 'cfg', 'config.yaml')
    trainer_cfg = OmegaConf.load(trainer_cfg_filename)
    merged_cfg = OmegaConf.merge(trainer_cfg, cfg)
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model = create_model(merged_cfg)
    signal.signal(signal.SIGINT, original_sigint_handler)
    main(model)


def fn(input, model):
    pr_input = preprocess_input(input)
    outputs = model(pr_input)
    output_img = draw(pr_input['q_img'][0], outputs[0])
    return output_img


def main(model):
    input_components = [
        ImageInput(
            shape=(320, 320),
            name='query',
        ),
        ImageInput(
            shape=(105, 105),
            name='s1',
        ),
        ImageInput(
            shape=(105, 105),
            name='s2',
        ),
        ImageInput(
            shape=(105, 105),
            name='s3',
        ),
    ]
    output_components = [
        ImageOutput(
            name='output'
        )
    ]
    widget = Interface(
        model=model,
        fn=fn,
        input_components=input_components,
        output_components=output_components
    )
    widget.launch()


if __name__ == '__main__':
    path = '/home/vladimir/data/omniglot/'
    run()
