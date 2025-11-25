import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from nte_aisdk.dynamic_few_shot import (
    DynamicFewShotDefaultSearchStrategy,
    DynamicFewShotExampleFieldMapping,
    DynamicFewShotExampleStore,
    DynamicFewShotModel,
    DynamicFewShotPromptField,
    DynamicFewShotPromptTemplate,
    DynamicFewShotSimpleExampleStore,
    DynamicFewShotWordSplitSearchStrategy,
)
from nte_aisdk.providers.azure import AzureProvider

load_dotenv()

# Create Azure provider
azure_provider = AzureProvider(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
)


# Define response schema for JSON output
class CategoryResponse(BaseModel):
    Class: str


def test_dynamic_few_shot_with_example_store():
    """Test DynamicFewShotModel with DynamicFewShotExampleStore (environment-based)"""
    print("=== Testing DynamicFewShotModel with DynamicFewShotExampleStore ===")

    # Create example store with environments
    example_store = DynamicFewShotExampleStore(
        host=os.getenv("ELASTICSEARCH_HOST"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD")),
        embedding_instance_configs=[{
            "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
            "api_key": os.getenv("AZURE_API_KEY"),
        }],
        index_prefix="sandbox--sdk-dev--examplestore",
    )

    # Create prompt template with text response
    prompt_template = DynamicFewShotPromptTemplate(
        instruction_text="You are an AI categorizer. Your task is to tag the comment with the correct label.",
        field_mapping=DynamicFewShotExampleFieldMapping(
            input="Comment",
            output="Class",
        ),
        num_examples=3,
        response_type="text",
    )

    # Create model
    model = DynamicFewShotModel(
        language_model=azure_provider.create_language_model(
            azure_deployment="gpt-4.1-mini",
            model_name="gpt-4.1-mini",
        ),
        prompt_template=prompt_template,
        example_store=example_store,
        search_strategy=DynamicFewShotDefaultSearchStrategy(),
    )

    # Test generation
    try:
        result = model.generate(query="Thank you for your help!", environment="staging")
        print(f"Generation result: {result}")
    except Exception as e:
        print(f"Generation failed: {e}")


def test_dynamic_few_shot_with_simple_example_store():
    """Test DynamicFewShotModel with DynamicFewShotSimpleExampleStore (space-based)"""
    print("\n=== Testing DynamicFewShotModel with DynamicFewShotSimpleExampleStore ===")

    # Create simple example store (space-based)
    simple_example_store = DynamicFewShotSimpleExampleStore(
        host=os.getenv("ELASTICSEARCH_HOST"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD")),
        embedding_model=azure_provider.create_embedding_model(
            azure_deployment="text-embedding-3-small",
            model_name="text-embedding-3-small",
        ),
        index_name="sandbox--sdk-dev--andrew-spaces-dfs2",
    )

    # Create prompt template with JSON response
    json_prompt_template = DynamicFewShotPromptTemplate(
        instruction_text="You are an AI categorizer. Your task is to tag the comment with the correct label and provide a confidence score.",
        field_mapping=DynamicFewShotExampleFieldMapping(
            input="primary_input_1",
            output="output_1",
        ),
        prompt_field=DynamicFewShotPromptField(
            input="Comment",
            output="Class",
        ),
        num_examples=2,
        response_type="json",
        response_schema=CategoryResponse,
    )

    # Create model with word split search strategy
    model = DynamicFewShotModel(
        language_model=azure_provider.create_language_model(
            azure_deployment="gpt-4.1-mini",
            model_name="gpt-4.1-mini",
        ),
        prompt_template=json_prompt_template,
        example_store=simple_example_store,
        search_strategy=DynamicFewShotWordSplitSearchStrategy(),
    )

    # Test generation with space_id
    try:
        result = model.generate(query="This product is amazing!", space_id="01BT6QKHSLVFKRKYLWLJHJLHZPRUW4I4PN")
        print(f"Generation result: {result}")
    except Exception as e:
        print(f"Generation failed: {e}")


def test_preview_generate():
    """Test preview generation functionality"""
    print("\n=== Testing Preview Generation ===")

    # Create example store with environments
    example_store = DynamicFewShotExampleStore(
        host=os.getenv("ELASTICSEARCH_HOST"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD")),
        embedding_instance_configs=[{
            "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
            "api_key": os.getenv("AZURE_API_KEY"),
        }],
        index_prefix="sandbox--sdk-dev--examplestore",
    )

    # Create prompt template
    prompt_template = DynamicFewShotPromptTemplate(
        instruction_text="Classify the sentiment of the given text.",
        field_mapping=DynamicFewShotExampleFieldMapping(
            input="Comment",
            output="Class",
        ),
        num_examples=2,
        response_type="text",
    )

    # Create model
    model = DynamicFewShotModel(
        language_model=azure_provider.create_language_model(
            azure_deployment="gpt-4.1-mini",
            model_name="gpt-4.1-mini",
        ),
        prompt_template=prompt_template,
        example_store=example_store,
        search_strategy=DynamicFewShotDefaultSearchStrategy(),
    )

    # Test preview generation with added examples
    added_examples = [
        {"Comment": "I love this!", "Class": "positive"},
        {"Comment": "This is terrible", "Class": "negative"},
    ]

    removed_ids = ["mFWyFJYBBlFhE9gV0ckg"]  # Replace with actual IDs if needed

    try:
        result = model.preview_generate(
            query="This is okay",
            added_examples=added_examples,
            removed_ids=removed_ids,
            environment="staging",
        )
        print(f"Preview generation result: {result}")
    except Exception as e:
        print(f"Preview generation failed: {e}")


def test_supplementary_inputs_with_simple_example_store():
    """Test dynamic few shot with supplementary inputs"""
    print("\n=== Testing Supplementary Inputs ===")

    # Create example store
    example_store = DynamicFewShotSimpleExampleStore(
        host=os.getenv("ELASTICSEARCH_HOST"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD")),
        embedding_model=azure_provider.create_embedding_model(
            azure_deployment="text-embedding-3-small",
            model_name="text-embedding-3-small",
        ),
        index_name="sandbox--sdk-dev--andrew-spaces-dfs2",
    )

    # Create prompt template with custom prompt field
    prompt_template = DynamicFewShotPromptTemplate(
        instruction_text="Classify the sentiment of the given text.",
        field_mapping=DynamicFewShotExampleFieldMapping(
            input="primary_input_1",
            output="output_1",
            Rating="supplementary_input_1",
        ),
        prompt_field=DynamicFewShotPromptField(
            input="Comment",
            output="Class",
            Rating="Rating",
        ),
        num_examples=4,
        response_type="json",
        response_schema=CategoryResponse
    )

    # Create model
    model = DynamicFewShotModel(
        language_model=azure_provider.create_language_model(
            azure_deployment="gpt-4.1-mini",
            model_name="gpt-4.1-mini",
        ),
        prompt_template=prompt_template,
        example_store=example_store,
        search_strategy=DynamicFewShotDefaultSearchStrategy(),
    )

    # Test generation with supplementary inputs
    supplementary_inputs = {"Rating": 100}

    result = model.generate(
        query="I need help with my account",
        supplementary_inputs=supplementary_inputs,
        space_id="01BT6QKHXK2RLLMBTFQBFY4IPINXW23BXA",
    )
    print(f"Generation with supplementary inputs result: {result}")

from nte_aisdk.providers.vertex import VertexProvider

vertex_provider = VertexProvider(
    location=os.environ["VERTEX_LOCATION"],
    project=os.environ["VERTEX_PROJECT"],
    credentials_base64=os.environ["VERTEX_CREDENTIALS_BASE64"],
)

def test_dfs_with_simple_example_store_vertex():
    """Test simple DFS with Vertex provider"""
    print("\n=== Testing DynamicFewShotModel with DynamicFewShotSimpleExampleStore with Vertex Provider ===")

    simple_example_store = DynamicFewShotSimpleExampleStore(
        host=os.getenv("ELASTICSEARCH_HOST"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD")),
        embedding_model=azure_provider.create_embedding_model(
            azure_deployment="text-embedding-3-small",
            model_name="text-embedding-3-small",
        ),
        index_name="sandbox--sdk-dev--andrew-spaces-dfs2",
    )

    # Create prompt template with JSON response
    json_prompt_template = DynamicFewShotPromptTemplate(
        instruction_text="You are an AI categorizer. Your task is to tag the comment with the correct label and provide a confidence score.",
        field_mapping=DynamicFewShotExampleFieldMapping(
            input="primary_input_1",
            output="output_1",
        ),
        prompt_field=DynamicFewShotPromptField(
            input="Comment",
            output="Class",
        ),
        num_examples=2,
        response_type="json",
        response_schema=CategoryResponse,
    )

    # Create model with word split search strategy
    model = DynamicFewShotModel(
        language_model=vertex_provider.create_language_model(
            model_name="gemini-2.5-flash",
        ),
        prompt_template=json_prompt_template,
        example_store=simple_example_store,
        search_strategy=DynamicFewShotWordSplitSearchStrategy(),
    )

    # Test generation with space_id
    try:
        result = model.generate(query="This product is amazing!", space_id="01BT6QKHSLVFKRKYLWLJHJLHZPRUW4I4PN")
        print(f"Generation result: {result}")
    except Exception as e:
        print(f"Generation failed: {e}")

def main():
    """Run all tests"""
    print("Starting Dynamic Few Shot v2 Testing...")

    # Test different configurations
    test_dynamic_few_shot_with_example_store()
    test_dynamic_few_shot_with_simple_example_store()
    test_preview_generate()
    test_supplementary_inputs_with_simple_example_store()
    test_dfs_with_simple_example_store_vertex()

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
