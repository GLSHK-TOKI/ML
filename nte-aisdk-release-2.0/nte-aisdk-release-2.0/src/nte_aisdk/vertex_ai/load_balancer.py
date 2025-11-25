import random

from google.genai import Client


class VertexAILoadBalancer:
    def __init__(self, instances: list[Client]) -> None:
        self.instances = instances

    def get_instance(self) -> Client:
        return random.choice(self.instances)