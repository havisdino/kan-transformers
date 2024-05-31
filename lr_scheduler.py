from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class RectifiedLinearLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        learning_rate_init: float,
        learning_rate_final: float,
        start_descending_step: int,
        stop_descending_step: int
    ):
        k = (
            (learning_rate_final - learning_rate_init)
            / (stop_descending_step - start_descending_step)
        )
        
        def schedule(step):
            if step <= start_descending_step:
                return learning_rate_init
            elif start_descending_step < step < stop_descending_step:
                return k * (step - start_descending_step) + learning_rate_init
            else:
                return learning_rate_final
        
        LambdaLR.__init__(self, optimizer, schedule)
