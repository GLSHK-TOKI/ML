import logging

from nte_aisdk import ElasticRateLimitStorage, RateLimiting
from nte_aisdk.access_control import AccessControl, ElasticAccessControlStorage
from nte_aisdk.knowledge_base import KnowledgeBaseChatModel, KnowledgeBaseStore
from nte_aisdk.providers.azure import AzureProvider


class AI:
    logger: logging.Logger
    store: KnowledgeBaseStore
    model: KnowledgeBaseChatModel
    rate_limit: RateLimiting
    access_control: AccessControl

    def init_app(self, app):
        azure_provider = AzureProvider(
            app.config["OPENAI_AZURE_ENDPOINT_1"],
            app.config["OPENAI_API_KEY_1"],
            app.config["OPENAI_LLM_API_VERSION"],
        )

        embedding_model = azure_provider.create_embedding_model(
            azure_deployment=app.config["OPENAI_EMBEDDING_AZURE_DEPLOYMENT"],
            model_name=app.config["OPENAI_EMBEDDING_AZURE_DEPLOYMENT"],
        )

        language_model = azure_provider.create_language_model(
            azure_deployment=app.config["OPENAI_LLM_AZURE_DEPLOYMENT"],
            model_name=app.config["OPENAI_LLM_AZURE_DEPLOYMENT"],
        )

        self.store = KnowledgeBaseStore(
            host=app.config["ELASTICSEARCH_HOST"],
            basic_auth=(
                app.config["ELASTICSEARCH_USERNAME"],
                app.config["ELASTICSEARCH_PASSWORD"],
            ),
            embedding_model=embedding_model,
            index_prefix=app.config["KNOWLEDGE_BASE_STORE_INDEX_PREFIX"],
        )

        self.model = KnowledgeBaseChatModel(
            language_model=language_model,
            store= self.store,
            retriever_size=30,
            retriever_threshold=1.2,
        )

        self.access_control = AccessControl(
            app.config["AZURE_TENANT_ID"],
            app.config["AZURE_CLIENT_ID"],
            app.config["AZURE_SHAREPOINT_DRIVE_ID"],
            app.config["AZURE_CLIENT_SECRET"],
            storage = ElasticAccessControlStorage(
                host=app.config["ELASTICSEARCH_HOST"],
                basic_auth=(
                    app.config["ELASTICSEARCH_USERNAME"],
                    app.config["ELASTICSEARCH_PASSWORD"],
                ),
                index_name=app.config["ACCESS_CONTROL_STORAGE_INDEX"],
            )
        )

        self.rate_limit = RateLimiting(
            storage=ElasticRateLimitStorage(
                host=app.config["ELASTICSEARCH_HOST"],
                basic_auth=(
                    app.config["ELASTICSEARCH_USERNAME"],
                    app.config["ELASTICSEARCH_PASSWORD"],
                ),
                index_name=app.config["RATE_LIMIT_STORAGE_INDEX"],
            ),
            config={"general-chatbot": {"window": "*/5 * * * *", "token_limit": 10000},
                    "knowledge-base-chat": {"window": "*/5 * * * *", "token_limit": 10000}
                    }
        )

ai = AI()