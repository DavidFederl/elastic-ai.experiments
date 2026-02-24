import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_inital_seed(seed: int, make_determenistic: bool) -> None:
    """Set the initial seed for the experiment.

    sets all seeding bases and decides if cuda should only use determensitic modules

    Args:
        seed (int): seed to set
        make_determenistic (bool): whether to make operations deterministic

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = make_determenistic
    torch.backends.cudnn.benchmark = not make_determenistic
    logger.info(f"Seed set to {seed}")
