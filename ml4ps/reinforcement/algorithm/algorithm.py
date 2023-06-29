from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(self, env, seed=None, val_env=None, test_env=None, run_dir=None,):
        pass

    @abstractmethod
    def learn(*args, logger=None, seed=None, batch_size=None, **kwargs):
        pass

    @abstractmethod
    def test(*args, test_env=None, res_dir=None, max_steps=None, **kwargs) -> float:
        pass

    def save_params(self, folder):
        pass

    def load_params(self, folder):
        pass

    def save(self, run_dir):
        pass
