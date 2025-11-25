import os

# Use os.environ['ENV_NAME'] for mandatory
# Use os.environ.get('ENV_NAME') for optional
# Use os.environ.get('ENV_NAME', 'DEFAULT') for optional with default value


class Local:
    DEBUG = True
    TESTING = True
    ELASTICSEARCH_HOST = os.environ.get("ELASTICSEARCH_HOST")
    ELASTICSEARCH_USERNAME = os.environ.get("ELASTICSEARCH_USERNAME")
    ELASTICSEARCH_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")
    ACCESS_CONTROL_STORAGE_INDEX = os.environ.get("ACCESS_CONTROL_STORAGE_INDEX")
    KNOWLEDGE_BASE_STORE_INDEX_PREFIX = os.environ.get("KNOWLEDGE_BASE_STORE_INDEX_PREFIX")
    RATE_LIMIT_STORAGE_INDEX = os.environ.get("RATE_LIMIT_STORAGE_INDEX")
    OPENAI_AZURE_ENDPOINT_1 = os.environ.get("OPENAI_AZURE_ENDPOINT_1")
    OPENAI_API_KEY_1 = os.environ.get("OPENAI_API_KEY_1")
    OPENAI_LLM_AZURE_DEPLOYMENT = os.environ.get("OPENAI_LLM_AZURE_DEPLOYMENT")
    OPENAI_LLM_API_VERSION = os.environ.get("OPENAI_LLM_API_VERSION")
    OPENAI_EMBEDDING_AZURE_DEPLOYMENT = os.environ.get("OPENAI_EMBEDDING_AZURE_DEPLOYMENT")
    OPENAI_EMBEDDING_API_VERSION = os.environ.get("OPENAI_EMBEDDING_API_VERSION")
    AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID")
    AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
    AZURE_SHAREPOINT_DRIVE_ID = os.environ.get("AZURE_SHAREPOINT_DRIVE_ID")
    AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")
    ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "http://localhost:3000")





class D0(Local):
    DEBUG = False
    TESTING = False


class T0(D0):
    pass


class P0(D0):
    pass
