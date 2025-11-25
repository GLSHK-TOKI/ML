import random

import google.auth.transport.requests
from google.oauth2 import service_account
from openai import OpenAI


class LlamaLoadBalancer:
    def __init__(self, instances: list[OpenAI], creds: list[service_account.Credentials]) -> None:
        self.instances = instances
        self.creds = creds

    def get_instance(self) -> OpenAI:
        # https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-credentials-refresher-class
        # Refresh Google credentials if they're not valid
        instance = random.choice(self.instances)
        cred = self.creds[self.instances.index(instance)]
        if not cred.valid:
            auth_req = google.auth.transport.requests.Request()
            cred.refresh(auth_req)

            if not cred.valid:
                msg = "Unable to refresh auth"
                raise RuntimeError(msg)

            instance.api_key = cred.token
        return instance