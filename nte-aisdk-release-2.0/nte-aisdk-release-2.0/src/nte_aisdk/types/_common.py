
import pydantic


class BaseModel(pydantic.BaseModel):
    """Base class for all models in NTE AI SDK."""

    model_config = pydantic.ConfigDict(
        protected_namespaces=(), # To allow field with "model_" prefix to be used.
    )
