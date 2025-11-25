import logging
from random import SystemRandom


class ProbabilityFilter(logging.Filter):
    cryptogen = SystemRandom()

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def filter(self, record):
        return self.cryptogen.random() < self.probability