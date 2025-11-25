
from nte_aisdk import types
from nte_aisdk.utils import ensure_arguments

from ...vector_store import VectorStore
from ._base_example_store import BaseExampleStore
from ._example_store_live import DynamicFewShotExampleStoreLive
from ._example_store_staging import DynamicFewShotExampleStoreStaging


class DynamicFewShotExampleStore(VectorStore, BaseExampleStore):
    """DynamicFewShotExampleStore is a class that extends BaseVectorStore for dynamic few shot usecases.

    Args:
        embedding_model: The embedding model to use for embedding operations
        index_prefix: The prefix name of the ElasticSearch index for this example store.
                Three indices should be created with the following names:
                    - {index_prefix}1
                    - {index_prefix}2
                    - {index_prefix}3
                    - {index_prefix}states
    """
    @property
    def uses_environments(self) -> bool:
        return True

    @ensure_arguments
    def __init__(
            self,
            host: str,
            basic_auth: str | tuple[str, str],
            embedding_instance_configs: list[types.AzureInstanceConfigWithoutAPIVersionOrDict],
            index_prefix: str,
        ):
        """Initialize the DynamicFewShotExampleStore with the given parameters.

        Args:
            host: The ElasticSearch host URL.
            basic_auth: The basic authentication credentials for ElasticSearch.
            embedding_instance_configs: Configurations with only the instance(s) of embedding
                model deployments (legacy style). Deployment name and API version are loaded
                from the state record in Elasticsearch, so they are not required here.
            index_prefix: The prefix name of the ElasticSearch index for this example store.
        """
        super().__init__(host, basic_auth)
        self.embedding_instance_configs = embedding_instance_configs
        self.index_prefix = index_prefix

        self.staging = DynamicFewShotExampleStoreStaging(self)
        self.live = DynamicFewShotExampleStoreLive(self)