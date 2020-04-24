from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    config = get_config(mode='train')
    video_loader, text_loader = get_loader(config.video_root_dir, config.mode, config.batch_size)
    solver = Solver(config, video_loader, text_loader)

    solver.build()
    solver.train()
