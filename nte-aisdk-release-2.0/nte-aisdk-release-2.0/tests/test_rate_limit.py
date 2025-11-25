import time
import unittest

from flask import Flask, g

from nte_aisdk.exception import SDKException
from nte_aisdk.rate_limit.rate_limit import RateLimiting


class MockRateLimitStorage:
    def __init__(self):
        self.data = {}

    def get(self, key, user_id):
        return self.data.get((key, user_id))

    def put(self, key, user_id, tokens, timestamp):
        self.data[(key, user_id)] = [{"_id": "new_id", "tokens": tokens, "timestamp": timestamp}]
        return self.data[(key, user_id)]

    def update(self, doc_id, tokens=None, timestamp=None):
        for v in self.data.values():
            if v[0]["_id"] == doc_id:
                if tokens is not None:
                    v[0]["tokens"] = tokens
                if timestamp is not None:
                    v[0]["timestamp"] = timestamp
                return v
        return None

class TestRateLimiting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app_context = self.app.app_context()
        self.app_context.push()

        self.storage = MockRateLimitStorage()
        self.rate_limiting = RateLimiting(self.storage)
        self.user_id = "test_user"
        self.key = "test_key"
        self.window = "* * * * *"
        self.default_token = 10
        self.token_limit = 10

        g.nte_aisdk_user = {"id": self.user_id}

    def tearDown(self):
        self.app_context.pop()

    async def test__check_new_record(self):
        result = await self.rate_limiting._check(self.user_id, self.key, self.window, self.token_limit)  # noqa: SLF001
        self.assertIsNotNone(result)

    async def test__hit(self):
        current_time = int(time.time() * 1000)
        self.storage.data[(self.key, self.user_id)] = [{"_id": "existing_id", "tokens": 5, "timestamp": current_time}]
        g.nte_aisdk_prompt_tokens = 2
        g.nte_aisdk_completion_tokens = 1
        await self.rate_limiting._hit(self.key) # noqa: SLF001
        data = self.storage.get(self.key, self.user_id)
        self.assertEqual(data[0]["tokens"], 2)

    async def test_get_token_balance(self):
        self.storage.data[(self.key, self.user_id)] = [{"tokens": 5}]
        balance = self.rate_limiting.get_token_balance(self.key, self.default_token)
        self.assertEqual(balance["tokens"], 5)

    async def test_get_token_balance_no_record(self):
        balance = self.rate_limiting.get_token_balance(self.key, self.default_token)
        self.assertEqual(balance["tokens"], 10)

    async def test_get_token_balance_multiple_records(self):
        self.storage.data[(self.key, self.user_id)] = [{"tokens": 5}, {"tokens": 10}]
        print(self.storage.data)
        with self.assertRaises(SDKException) as context:
            self.rate_limiting.get_token_balance(self.key, self.default_token)
        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(context.exception.message, "More than One Rate limiting Records have been found")

if __name__ == "__main__":
    unittest.main(verbosity=2)