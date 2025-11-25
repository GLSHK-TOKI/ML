from __future__ import annotations

import asyncio
import inspect
import json
import logging
from functools import wraps
from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.request import urlopen

import httpx
from azure.core.exceptions import HttpResponseError
from azure.identity.aio import OnBehalfOfCredential
from flask import current_app, g, request
from jose import jwt

from nte_aisdk.cache import get_cache
from nte_aisdk.errors import AuthError, SDKError
from nte_aisdk.utils import ensure_arguments, is_inside_running_loop

if TYPE_CHECKING:
    from .access_control_storage import AccessControlStorage

logger = logging.getLogger(__name__)

CACHE_KEY_NAMESPACE = "nte_aisdk_access"

class AccessControl:
    """KnowledgeBaseAccessControl is a class that can be used to validate access to knowledge base.
    The access control is based on the user's access on SharePoint folder.
    """

    tenant_id: str
    client_id: str
    drive_id: str
    client_secret: str
    _storage: AccessControlStorage | None

    @ensure_arguments
    def __init__(
            self,
            tenant_id: str,
            client_id: str,
            drive_id: str,
            client_secret: str,
            storage: AccessControlStorage | None = None,
            *,
            enable_cache: bool = False,
        ):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.drive_id = drive_id
        self.client_secret = client_secret
        self.scopes = ["https://graph.microsoft.com/.default"]
        self._storage = storage
        self._enable_cache = enable_cache

    def token_required(self, f):
        if inspect.iscoroutinefunction(f):
            @wraps(f)
            async def async_decorated_function(*args, **kwargs):
                """This function is a decorator that retrieve the access token from request header and
                storage in flask context.
                """
                self._authenticate_request()

                return await f(*args, **kwargs)
            return async_decorated_function
        @wraps(f)
        def decorated_function(*args, **kwargs):
            """Synchronous version of token_required for streaming endpoints."""
            self._authenticate_request()

            return f(*args, **kwargs)
        return decorated_function

    async def filter_granted_collections(self, collections) -> list[dict[str, str]]:
        if len(collections) == 0:
            return []
        token = await self._get_delegated_token()

        if not token:
            logger.error("Error acquiring token with delegated flow")
            return []

        access_granted = await self._get_access_granted_responses(collections, token)
        return [
            {
                **item,
                "webUrl": response["webUrl"],
                "name": response["name"]
            }
            for item in collections
            for response in access_granted
            if item["driveItemId"] == response["id"]
        ]

    async def get_granted_collections(self) -> list[dict[str, str]]:
        if self._storage is None:
            msg = "AccessControlStorage is required to get granted collections"
            raise SDKError(msg)

        collections = self._storage.list_collections()
        if len(collections) == 0:
            return []
        token = await self._get_delegated_token()

        if not token:
            logger.error("Error acquiring token with delegated flow")
            return []

        access_granted = await self._get_access_granted_responses(collections, token)
        return [
            item
            for item in collections
            if any(
                response["id"] == item["driveItemId"]
                for response in access_granted
            )
        ]

    @ensure_arguments
    async def has_access(self, collection_id: str) -> bool:
        token = await self._get_delegated_token()

        if token:
            if self._enable_cache:
                cache = get_cache()
                cache_key = f"{CACHE_KEY_NAMESPACE}:{collection_id}:{g.nte_aisdk_user['id']}"

                if cache_key in cache:
                    logger.debug(f"Cache hit for collection_id={collection_id} user_id={g.nte_aisdk_user['id']}")
                    return cache[cache_key]["has_access"]

                logger.debug(f"Cache miss for collection_id={collection_id} user_id={g.nte_aisdk_user['id']}")

            # Set the API endpoint
            endpoint = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/items/{collection_id}"
            # Make the GET request
            headers = {"Authorization": "Bearer " + token}
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint, headers=headers, timeout=None)
                logger.info(f"Graph API: user_id={g.nte_aisdk_user['id']} status={response.status_code} count=1")
            if response.status_code == HTTPStatus.OK:
                logger.debug(f"user_id={g.nte_aisdk_user['id']} has access to collection_id={collection_id}")
                access_result = {
                    "has_access": True,
                    "content": response.json()
                }
            else:
                logger.debug(f"user_id={g.nte_aisdk_user['id']} has no access to collection_id={collection_id}")
                access_result = {
                    "has_access": False
                }

            if self._enable_cache:
                cache[cache_key] = access_result
            return access_result["has_access"]

        logger.error(f"Error acquiring delegated token for Graph API during access check. user_id={g.nte_aisdk_user['id']}")
        return False

    def enforce_access(self, collection_id_key: str):
        """This function is a decorator that rejects requests with access denied on the collection
        with the given key of collection_id.

        Args:
            collection_id_key (str): The key used to retrieve the collection ID from the request JSON.

        Returns:
            function: The decorated function.

        Raises:
            AuthError: If access is denied to the collection with the given ID.
        """
        def decorator(f):
            if is_inside_running_loop() or inspect.iscoroutinefunction(f):
                @wraps(f)
                async def async_decorated_function(*args, **kwargs):
                    # Get the collection ID from URL parameters
                    collection_id = kwargs.get(collection_id_key)
                    # if not found in URL, try the request body
                    if not collection_id:
                        collection_id = request.json.get(collection_id_key)
                    if await self.has_access(collection_id):
                        if inspect.iscoroutinefunction(f):
                            return await f(*args, **kwargs)
                        return f(*args, **kwargs)

                    msg = f"Access denied to the collection with id: {collection_id}"
                    raise AuthError(msg)

                return async_decorated_function
            @wraps(f)
            def decorated_function(*args, **kwargs):
                """Synchronous version of enforce_access for streaming endpoints."""
                # Get the collection ID from URL parameters
                collection_id = kwargs.get(collection_id_key)
                # if not found in URL, try the request body
                if not collection_id:
                    collection_id = request.json.get(collection_id_key)

                # Run the async has_access check in a sync context
                if asyncio.run(self.has_access(collection_id)):
                    return f(*args, **kwargs)

                msg = f"Access denied to the collection with id: {collection_id}"
                raise AuthError(msg)

            return decorated_function
        return decorator

    def _authenticate_request(self):
        """This function is a decorator that retrieves the access token from request header, extracts user info and
        stores them in flask context.
        """
        token = self._get_token_auth_header()

        payload = self._verify_and_decode_token(token)

        user_id = payload.get("oid")
        email = payload.get("upn") or payload.get("preferred_username")

        g.nte_aisdk_user = {"id": user_id, "email": email}

        g.nte_aisdk_access_token = token

    async def _get_access_granted_responses(self, collections, token):
        if self._enable_cache:
            # Check for cache hit and exclude them from batch call
            cache = get_cache()

            cached_responses = [] # to store responses that are already in cache, which has access
            cached_collections = [] # to store collections that are already in cache
            not_cached_collections = [] # to store collections that are not in cache
            for collection in collections:
                collection_id = collection["driveItemId"]
                cache_key = f"{CACHE_KEY_NAMESPACE}:{collection_id}:{g.nte_aisdk_user['id']}"
                if cache_key in cache:
                    cached_collections.append(collection)
                    if cache[cache_key]["has_access"]:
                        cached_responses.append(cache[cache_key]["content"])
                else:
                    not_cached_collections.append(collection)

            if len(cached_collections) > 0:
                logger.debug(f"Cache hit for collection_ids=[{','.join([collection['driveItemId'] for collection in cached_collections])}] user_id={g.nte_aisdk_user['id']}")
            if len(not_cached_collections) > 0:
                logger.debug(f"Cache miss for collection_ids=[{','.join([collection['driveItemId'] for collection in not_cached_collections])}] user_id={g.nte_aisdk_user['id']}")

        # Graph API has a limit of 20 items per request
        collections_to_process = not_cached_collections if self._enable_cache else collections
        chunk_size = 20
        chunks = [
            collections_to_process[i:i + chunk_size]
            for i in range(0, len(collections_to_process), chunk_size)
        ]
        ok_responses = []
        for chunk in chunks:
            payload = {
                "requests": [
                    {
                        "id": itemId["driveItemId"],
                        "method": "GET",
                        "url": f"/drives/{self.drive_id}/items/{itemId['driveItemId']}",
                        "headers": {
                            "Content-Type": "application/json",
                        },
                    }
                    for itemId in chunk
                ]
            }
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    "https://graph.microsoft.com/v1.0/$batch",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                logger.info(f"Graph API: user_id={g.nte_aisdk_user['id']} status={res.status_code} count={len(chunk)}")

                result = res.json()
                if "responses" in result:
                    for response in result["responses"]:
                        if "id" not in response or "status" not in response:
                            logger.warning("Invalid batch response in access check: %s", response)
                            continue

                        collection_id = response["id"]
                        cache_key = f"{CACHE_KEY_NAMESPACE}:{collection_id}:{g.nte_aisdk_user['id']}"
                        if response["status"] == HTTPStatus.OK:
                            if self._enable_cache:
                                cache[cache_key] = {
                                    "has_access": True,
                                    "content": response["body"]
                                }
                            ok_responses.append(response["body"])
                        elif self._enable_cache:
                            cache[cache_key] = {
                                "has_access": False,
                            }
        if self._enable_cache:
            return ok_responses + cached_responses
        return ok_responses

    def _get_token_auth_header(self) -> str:
        """Obtains the Access Token from the Authorization Header
        """
        auth = request.headers.get("Authorization", None)
        if not auth:
            msg = "Authorization header is expected"
            raise AuthError(msg)

        parts = auth.split()

        if parts[0].lower() != "bearer":
            msg = "Authorization header must start with Bearer"
            raise AuthError(msg)
        if len(parts) == 1:
            msg = "Token not found"
            raise AuthError(msg)
        max_parts = 2
        if len(parts) > max_parts:
            msg = "Authorization header must be Bearer token"
            raise AuthError(msg)

        token = parts[1]
        return token  # noqa: RET504

    async def _get_delegated_token(self) -> str:
        # Define your Azure app credentials
        tenant_id = self.tenant_id
        client_id = self.client_id
        client_secret = self.client_secret
        user_access_token = g.nte_aisdk_access_token

        # Initialize the OnBehalfOfCredential
        credential = OnBehalfOfCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            user_assertion=user_access_token
        )

        # Acquire a token for the Graph API
        try:
            access_token = await credential.get_token("https://graph.microsoft.com/.default")
        except HttpResponseError as e:
            msg = f"Error acquiring token: {e}"
            raise AuthError(msg) from e
        token = access_token.token
        await credential.close()
        return token

    def _verify_and_decode_token(self, token: str) -> dict:
        """Determine if the token is valid and return the decoded token
        """
        azure_client_id = current_app.config.get("AZURE_CLIENT_ID", None)
        azure_tenant_id = current_app.config.get("AZURE_TENANT_ID", None)
        azure_valid_issuers = [
            f"https://sts.windows.net/{azure_tenant_id}/",
            f"https://login.microsoftonline.com/{azure_tenant_id}/v2.0",
        ]
        azure_valid_audiences = [f"api://{azure_client_id}", str(azure_client_id)]

        if not azure_client_id or not azure_tenant_id:
            msg = f"Client ID and Tenant ID not found in backside configuration: {azure_client_id}, {azure_tenant_id}"
            raise AuthError(msg)

        try:
            jsonurl = urlopen("https://login.microsoftonline.com/" +
                            azure_tenant_id + "/discovery/v2.0/keys")
            jwks = json.loads(jsonurl.read())
            unverified_header = jwt.get_unverified_header(token)
            unverified_claims = jwt.get_unverified_claims(token)
            issuer = unverified_claims.get("iss")
            audience = unverified_claims.get("aud")
            rsa_key = {}
            for key in jwks["keys"]:
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = {
                        "kty": key["kty"],
                        "kid": key["kid"],
                        "use": key["use"],
                        "n": key["n"],
                        "e": key["e"]
                    }
        except Exception as err:
            msg = "Unable to parse authentication token."
            raise AuthError(msg) from err

        if issuer not in azure_valid_issuers:
            msg = f"Issuer {issuer} not in {','.join(azure_valid_issuers)}"
            raise AuthError(msg)

        if audience not in azure_valid_audiences:
            msg = f"Audience {audience} not in {','.join(azure_valid_audiences)}"
            raise AuthError(msg)
        if rsa_key:
            try:
                payload = jwt.decode(
                    token,
                    rsa_key,
                    algorithms=["RS256"],
                    audience=audience,
                    issuer=issuer
                )
            except jwt.ExpiredSignatureError as signatureerr:
                msg = "Token is expired"
                raise AuthError(msg) from signatureerr
            except jwt.JWTClaimsError as jwterror:
                msg = "Incorrect claims, please check the audience and issuer"
                raise AuthError(msg) from jwterror
            except Exception as err:
                msg = "Unable to parse authentication token."
                raise AuthError(msg) from err
            return payload
        msg = "Unable to find appropriate key."
        raise AuthError(msg)
