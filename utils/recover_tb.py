import re
import os
import sys
from torch.utils.tensorboard import SummaryWriter


def parse_log_file(path_to_file, writer):
    trn_pattern = re.compile(r'Step - (\d+) loss - ([\d.]+)')
    vld_pattern = re.compile(r'([\w_/]+) - ([\d.]+)')
    cur_step = 0
    with open(path_to_file, 'r') as file:
        for line in file:
            trn_match = trn_pattern.findall(line)
            if trn_match:
                trn_match = trn_match[0]
                cur_step = int(trn_match[0])
                cur_loss = float(trn_match[1])
                writer.add_scalar('trn/loss', cur_loss, global_step=cur_step)
                continue
            vld_match = vld_pattern.findall(line)
            if vld_match:
                for match in vld_match:
                    metric_name = match[0]
                    metric_value = float(match[1])
                    writer.add_scalar(metric_name, metric_value, global_step=cur_step)


def run(path_to_file):
    path_to_save = os.path.split(path_to_file)[0]
    writer = SummaryWriter(log_dir=path_to_save)
    parse_log_file(path_to_file, writer)
    writer.close()


if __name__ == '__main__':
    path_to_file = sys.argv[1]
    run(path_to_file)
