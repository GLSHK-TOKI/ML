from __future__ import annotations

import asyncio
import inspect
import logging
import time
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING

from croniter import croniter
from flask import g
from pytz import utc

from nte_aisdk import errors
from nte_aisdk.utils import ensure_arguments, is_inside_running_loop

if TYPE_CHECKING:
    from .rate_limit_storage import RateLimitStorage

logger = logging.getLogger(__name__)

class RateLimiting:
    """RateLimiting is a class that manages rate limiting for individual user for AI SDK calls,
    using a vector store for record storage.

    Args:
            config (dict): The configuration for the reset window and token limit for each key. The format is:
            {
                "key_name": {
                    "window": "cron_expression",
                    "token_limit": int,
                    "type": "point" | "token"
                }, ...
            }
    """
    @ensure_arguments
    def __init__(
            self,
            storage: RateLimitStorage,
            config: dict[str, dict[str, str|int]]
        ):
        self._validate_config(config)
        self._storage = storage
        self._config = config

    def limit(self, key: str, points: int=0):
        """Decorator to limit the rate of the function call for the user in request.

        This function will check the token balance of the user and the expiration of the window.
        If the rate limit is reached, it will throw a RateLimitingError.
        Else, it will continue to the function and consume the token.

        Args:
            key (str): The key to identify the rate limit. Use to separate different rate limits.
            points (int): The number of points consumed for calling the model.

        Returns:
            decorator: The decorator function that can be applied to other functions.
        """
        def decorator(func):
            if is_inside_running_loop() or inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    # Check and consume the token
                    await self._check(g.nte_aisdk_user["id"], key)
                    if inspect.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    await self._hit(key, points)
                    return result
                return async_wrapper
            @wraps(func)
            def wrapper(*args, **kwargs):
                """Synchronous version of the decorator to limit the rate of a function call."""
                # Run the async check in a sync context
                asyncio.run(self._check(g.nte_aisdk_user["id"], key))

                # Execute the decorated function
                result = func(*args, **kwargs)

                # When the result is a generator (e.g., streaming response),
                # We need to wrap it in another generator that will call _hit after the stream is exhausted.
                def result_generator():
                    try:
                        yield from result
                    finally:
                        # Run the async hit in a sync context after the stream is done
                        asyncio.run(self._hit(key, points))
                return result_generator()
            return wrapper

        return decorator

    @ensure_arguments
    def get_token_balance(self, key: str):
        """Get the token balance of the user for the given key.

        Args:
            key (str): The key to identify the rate limit. Use to separate different rate limits.

        Returns:
            int: The token balance of the user for the given key.
        """
        if self._config.get(key) is None:
            msg = "Rate limiting reset window and token limit config settings have not been found"
            raise errors.SDKError(msg)

        rate_limit_record = self._storage.get(key, g.nte_aisdk_user["id"])
        if rate_limit_record:
            return {
                "tokens": rate_limit_record["tokens"],
                "timestamp": self._get_next_token_reset_time(self._config.get(key))
            }

        # If no record found, return the token limit
        return {
            "tokens": self._config[key]["token_limit"],
            "timestamp": self._get_next_token_reset_time(self._config.get(key))
        }

    async def _check(self, user_id: str, key: str):
        # Query ElasticSearch
        rate_limit_record = self._storage.get(key, user_id)
        current_time = int(time.time() * 1000)  # Current time in ms

        if rate_limit_record:
            if self._config.get(key) is None:
                msg = "Rate limiting reset window and token limit config settings have not been found"
                raise errors.SDKError(msg)
            if self._is_window_expired(rate_limit_record["timestamp"], str(self._config[key]["window"])):
                # If the window is expired, UPDATE the record by using doc_id
                return await self._reset(rate_limit_record["_id"], current_time, int(self._config[key]["token_limit"]))

            if rate_limit_record["tokens"] < 1:
                msg = "Rate limit reached"
                raise errors.RateLimitError(msg)
            # If the window is not expired, CONSUME the token by using user_id with token used
            return rate_limit_record

        # If no record found and no rate limit config found, raise an error
        if self._config.get(key) is None:
            msg = "Rate limit Key config not found"
            raise errors.SDKError(msg)
        # If no record found, CREATE a new record
        return await self._create_limit(key, user_id, current_time, int(self._config[key]["token_limit"]))

    def _is_window_expired(self, timestamp: float, window: str):
        cron = croniter(window, datetime.fromtimestamp(timestamp / 1000, tz=utc))
        next_time = cron.get_next(datetime)
        return datetime.now(tz=utc) > next_time

    async def _create_limit(self, key: str, user_id: str, current_time: float, token_limit: int):
        return self._storage.put(key, user_id, token_limit, current_time)

    async def _reset(self, es_doc_id: str, current_time: float, token_limit: int):
        return self._storage.update(
            doc_id=es_doc_id,
            tokens=token_limit,
            timestamp=current_time
        )

    async def _hit(self, key: str, points: int):
        rate_limit_record = self._storage.get(key, g.nte_aisdk_user["id"])
        if rate_limit_record:
            if "type" not in self._config[key]:
                self._config[key]["type"] = "token"
            match self._config[key]["type"]:
                case "token":
                    prompt_tokens = g.get("nte_aisdk_prompt_tokens", 0)
                    completion_tokens = g.get("nte_aisdk_completion_tokens", 0)
                    token_remained = rate_limit_record["tokens"] - prompt_tokens - completion_tokens
                    result = self._storage.update(doc_id=rate_limit_record["_id"], tokens=token_remained)
                    logger.debug(result)
                    return result
                case "point":
                    prompt_points = points
                    if prompt_points is None:
                        msg = "Value of consumed points is not defined"
                        raise errors.InvalidArgumentError(msg)
                    points_remained = rate_limit_record["tokens"] - prompt_points
                    result = self._storage.update(
                        doc_id=rate_limit_record["_id"],
                        tokens=points_remained
                    )
                    logger.debug(result)
                    return result
                case _:
                    msg = "Value passed for rate limiting type is incorrect"
                    raise errors.InvalidArgumentError(msg)
        return rate_limit_record

    def _get_next_token_reset_time(self, mapping_key_config):
        # Convert the timestamp to a datetime object
        cron_expression = mapping_key_config["window"]

        # Get the current time in UTC
        current_time = datetime.now(tz=utc)

        # Get the next scheduled time from the cron expression
        cron = croniter(cron_expression, current_time)
        next_time_from_now = cron.get_next(datetime)
        return int(next_time_from_now.timestamp() * 1000)

    def _validate_config(self, config: dict[str, dict[str, str | int]]) -> None:
        if not isinstance(config, dict):
            msg = "Rate limiting reset window and token limit config must be a dictionary"
            raise errors.InvalidArgumentError(msg)

        for key, value in config.items():
            if not isinstance(key, str):
                msg = f"Rate limiting reset window and token limit config key must be a string, got {type(key).__name__}"
                raise errors.InvalidArgumentError(msg)
            if not isinstance(value, dict):
                msg = f"Rate limiting reset window and token limit config value for key '{key}' must be a dictionary, got {type(value).__name__}"
                raise errors.InvalidArgumentError(msg)

            if "window" not in value or "token_limit" not in value:
                msg = f"Config for key '{key}' must contain 'window' and 'token_limit' keys"
                raise errors.InvalidArgumentError(msg)

            for sub_key, sub_value in value.items():
                if not isinstance(sub_key, str):
                    msg = f"Rate limiting reset window and token limit config sub-key for key '{key}' must be a string, got {type(sub_key).__name__}"
                    raise errors.InvalidArgumentError(msg)
                if not isinstance(sub_value, str | int):
                    msg = f"Rate limiting reset window and token limit config sub-value for key '{key}' must be a string or int, got {type(sub_value).__name__}"
                    raise errors.InvalidArgumentError(msg)
