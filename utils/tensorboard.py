import importlib
from typing import Optional, Callable
from datetime import datetime

class TensorboardWriter:
    """
    Log metrics into a directory for visualization within the TensorBoard.

    Parameters
    ----------
    log_dir : str, optional
        Paht to the folder to save logs for TensorBoard

    enabled : bool, optional, default=False
        Enable TensorBoard or not
    """

    def __init__(
        self, log_dir: Optional[str] = None, enabled: bool = False
    ):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # retrieve vizualization writer
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                        "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                        "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the config file."
                print(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step: int, mode: str = 'train') -> None:
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_second', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name: str) -> Callable:
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        # default action for returning methods defined in this class, set_step() for instance
        else:
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("Type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
