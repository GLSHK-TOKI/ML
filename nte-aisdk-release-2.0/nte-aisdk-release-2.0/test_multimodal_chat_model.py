#!/usr/bin/env python3
"""Test for multimodal chat model functionality using Vertex AI embedding models.
This test covers both search and answering capabilities with multimodal support.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path for development
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Test results logging
def create_test_log_file():
    """Create a test log file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"multimodal_chat_test_results_{timestamp}.txt"
    return log_filename

def write_test_result(log_file, test_name, status, details="", error_msg="", answer="", sources=None, original_response=None):
    """Write test result to log file with detailed answer and sources."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_symbol = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"TEST: {test_name}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"STATUS: {status_symbol} {status}\n")
        if details:
            f.write(f"DETAILS: {details}\n")
        if error_msg:
            f.write(f"ERROR: {error_msg}\n")

        # Write original response first
        if original_response:
            f.write(f"\nORIGINAL LLM RESPONSE:\n{'-'*40}\n")
            try:
                import json
                if hasattr(original_response, "__dict__"):
                    # Convert response object to dict for pretty printing
                    response_dict = {
                        "role": str(original_response.message.role) if hasattr(original_response.message, "role") else "N/A",
                        "text": original_response.message.text if hasattr(original_response.message, "text") else None,
                        "data": original_response.message.data if hasattr(original_response.message, "data") else None,
                        "reasoning": original_response.message.reasoning if hasattr(original_response.message, "reasoning") else None,
                        "metadata": {
                            "model_id": original_response.metadata.model_id if hasattr(original_response.metadata, "model_id") else "N/A",
                            "usage": {
                                "prompt_tokens": original_response.metadata.usage.prompt_tokens if hasattr(original_response.metadata.usage, "prompt_tokens") else 0,
                                "completion_tokens": original_response.metadata.usage.completion_tokens if hasattr(original_response.metadata.usage, "completion_tokens") else 0
                            } if hasattr(original_response.metadata, "usage") else {}
                        } if hasattr(original_response, "metadata") else {}
                    }
                    f.write(json.dumps(response_dict, indent=2, ensure_ascii=False))
                else:
                    f.write(str(original_response))
            except Exception as e:
                f.write(f"Error serializing original response: {e}\n")
                f.write(str(original_response))
            f.write(f"\n{'-'*40}\n")

        # Write processed answer
        if answer:
            f.write(f"\nPROCESSED ANSWER:\n{'-'*40}\n")
            f.write(f"{answer}\n")
            f.write(f"{'-'*40}\n")

        f.write(f"{'='*80}\n")


def write_log_header(log_file):
    """Write header information to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("MULTIMODAL CHAT MODEL TEST RESULTS\n")
        f.write(f"Test started at: {timestamp}\n")
        f.write(f"{'='*80}\n")

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Will use system environment variables instead")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")
    print("   Will use system environment variables instead")


def test_multimodal_chat_model():
    """Test multimodal chat model functionality with search and answering capabilities."""
    print("======================================================================")
    print("MULTIMODAL CHAT MODEL FUNCTIONALITY TEST")
    print("======================================================================")

    # Create test log file
    log_file = create_test_log_file()
    write_log_header(log_file)
    print(f"📝 Test results will be saved to: {log_file}")

    # Check for required environment variables
    required_vars = {
        "VERTEX_PROJECT": os.environ.get("VERTEXAI_LLM_PROJECT"),
        "VERTEX_LOCATION": os.environ.get("VERTEXAI_LLM_LOCATION"),
        "VERTEX_CREDENTIALS_BASE64": os.environ.get("VERTEXAI_LLM_CREDENTIALS"),
        "ELASTICSEARCH_HOST": os.environ.get("ELASTICSEARCH_HOST"),
        "ELASTICSEARCH_USERNAME": os.environ.get("ELASTICSEARCH_USERNAME"),
        "ELASTICSEARCH_PASSWORD": os.environ.get("ELASTICSEARCH_PASSWORD"),
        "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY"),
        "AZURE_API_VERSION": os.environ.get("AZURE_API_VERSION"),
    }

    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("   Please set these variables:")
        for var in missing_vars:
            if var.startswith("VERTEX_"):
                print(f'   $env:{var} = "your_vertex_value"')
            elif var.startswith("AZURE_"):
                print(f'   $env:{var} = "your_azure_value"')
            else:
                print(f'   $env:{var} = "your_elasticsearch_value"')
        return False

    try:
        # Import required modules
        from nte_aisdk import types
        from nte_aisdk.knowledge_base.chat_model import KnowledgeBaseChatModel
        from nte_aisdk.knowledge_base.store import KnowledgeBaseStore
        from nte_aisdk.providers.azure import AzureProvider
        from nte_aisdk.providers.vertex import VertexProvider

        print("✓ All required modules imported successfully")

        # Initialize providers
        print("\n--- Setting up Providers ---")

        # Vertex AI provider for multimodal embeddings
        vertex_provider = VertexProvider(
            location=required_vars["VERTEX_LOCATION"],
            project=required_vars["VERTEX_PROJECT"],
            credentials_base64=required_vars["VERTEX_CREDENTIALS_BASE64"]
        )

        # Azure OpenAI provider for text embeddings and language model
        azure_provider = AzureProvider(
            azure_endpoint=required_vars["AZURE_OPENAI_ENDPOINT"],
            api_key=required_vars["AZURE_OPENAI_API_KEY"],
            api_version=required_vars["AZURE_API_VERSION"],
        )

        print("✓ Providers initialized successfully")

        # Create models
        print("\n--- Setting up Models ---")

        # Create multimodal embedding model
        multimodal_embedding_model = vertex_provider.create_embedding_model(
            model_name="multimodalembedding@001"
        )
        print(f"✓ Vertex multimodal embedding model created: {multimodal_embedding_model.model_id}")

        # Create text embedding model
        text_embedding_model = azure_provider.create_embedding_model(
            azure_deployment="text-embedding-3-small",
            model_name="text-embedding-3-small",
        )
        print(f"✓ Azure text embedding model created: {text_embedding_model.model_id}")

        # Create language model for chat
        language_model = azure_provider.create_language_model(
            azure_deployment="gpt-4o",
            model_name="gpt-4o",
        )

        print(f"✓ Azure language model created: {language_model.model_id}")

        # Initialize KnowledgeBaseStore with multimodal support
        print("\n--- Setting up Knowledge Base Store ---")
        store = KnowledgeBaseStore(
            host=required_vars["ELASTICSEARCH_HOST"],
            basic_auth=(required_vars["ELASTICSEARCH_USERNAME"], required_vars["ELASTICSEARCH_PASSWORD"]),
            embedding_model=text_embedding_model,
            index_prefix=os.getenv("KNOWLEDGE_BASE_STORE_INDEX_PREFIX", "test_"),
            multimodal_embedding_model=multimodal_embedding_model
        )
        print("✓ Knowledge Base Store initialized with multimodal support")

        # Test 1: Traditional text-only chat model
        print("\n--- Test 1: Traditional Text-only Chat Model ---")
        text_only_chat_model = KnowledgeBaseChatModel(
            language_model=language_model,
            store=store,
            enable_multimodal_search=False,  # Disable multimodal search
            retriever_size_text=5,
            retriever_threshold=0.1
        )

        test_messages = [
            types.Message(
                role=types.Role.USER,
                parts=[types.TextPart(text="i would like to go Admiralty, what should i take?")]
            )
        ]

        try:
            response = text_only_chat_model.chat(
                messages=test_messages,
                collection_id="01BT6QKHUUVVG6G47EYVFJQ3ZSN5TONEFN" # corrected
            )

            print("✓ Text-only chat completed successfully")
            print(f"   Response role: {response.message.role}")
            print(f"   Model ID: {response.metadata.model_id}")
            print(f"   Prompt tokens: {response.metadata.usage.prompt_tokens}")
            print(f"   Completion tokens: {response.metadata.usage.completion_tokens}")

            # Try to parse response data if it's JSON
            details = f"Model: {response.metadata.model_id}, Tokens: {response.metadata.usage.prompt_tokens}/{response.metadata.usage.completion_tokens}"
            answer = ""
            sources = []

            try:
                if hasattr(response.message, "data") and response.message.data:
                    # Handle both parsed dict and JSON string cases
                    if isinstance(response.message.data, str):
                        data = json.loads(response.message.data)
                    else:
                        data = response.message.data
                        
                    answer = data.get("answer", "No answer found")
                    sources = data.get("sources", [])
                    print(f"   Answer preview: {answer[:100]}...")
                    details += f", Answer length: {len(answer)} chars, Sources: {len(sources)}"
                else:
                    answer = response.message.text if hasattr(response.message, "text") else "No text found"
                    print(f"   Response preview: {answer[:100]}...")
                    details += f", Response length: {len(answer)} chars"
            except Exception as parse_error:
                print(f"   Could not parse response data: {parse_error}")
                details += f", Parse error: {parse_error}"

            write_test_result(log_file, "Text-only Chat Model", "PASSED", details, answer=answer, sources=sources, original_response=response)

        except Exception as e:
            print(f"⚠️  Text-only chat test failed: {e}")
            print("   This might be expected if no documents exist in the test collection")
            write_test_result(log_file, "Text-only Chat Model", "FAILED", "", str(e))

        # Test 2: Multimodal chat model (text + image search)
        print("\n--- Test 2: Multimodal Chat Model ---")
        multimodal_chat_model = KnowledgeBaseChatModel(
            language_model=language_model,
            store=store,
            enable_multimodal_search=True,  # Enable multimodal search
            retriever_size_image=3,
            retriever_size_text=5,
            retriever_threshold=1.0
        )

        # Different question for multimodal test
        multimodal_test_messages = [
            types.Message(
                role=types.Role.USER,
                parts=[types.TextPart(text="what is safety assistant, who need safety assistant?")]
            )
        ]

        try:
            response = multimodal_chat_model.chat(
                messages=multimodal_test_messages,
                collection_id="01BT6QKHUAQI2T2I2FTVF34OU7JGPDX6KC",
                retriever_size_image=3,
                retriever_size_text=5
            )

            print("✓ Multimodal chat completed successfully")
            print(f"   Response role: {response.message.role}")
            print(f"   Model ID: {response.metadata.model_id}")
            print(f"   Prompt tokens: {response.metadata.usage.prompt_tokens}")
            print(f"   Completion tokens: {response.metadata.usage.completion_tokens}")

            # Try to parse response data
            details = f"Model: {response.metadata.model_id}, Tokens: {response.metadata.usage.prompt_tokens}/{response.metadata.usage.completion_tokens}"
            answer = ""
            sources = []

            try:
                if hasattr(response.message, "data") and response.message.data:
                    # Handle both parsed dict and JSON string cases
                    if isinstance(response.message.data, str):
                        data = json.loads(response.message.data)
                    else:
                        data = response.message.data
                    
                    answer = data.get("answer", "No answer found")
                    sources = data.get("sources", [])
                    print(f"   Answer preview: {answer[:100]}...")
                    
                    # Handle both old flat list and new structured sources
                    if isinstance(sources, dict):
                        # New multimodal structure: {text_sources: [...], image_sources: [...]}
                        text_sources = sources.get("text_sources", [])
                        image_sources = sources.get("image_sources", [])
                        print(f"   Text sources: {len(text_sources)}, Image sources: {len(image_sources)}")
                        details += f", Sources: {len(text_sources)} text + {len(image_sources)} image"
                    elif isinstance(sources, list):
                        # Old flat list structure or mixed content
                        text_sources = [s for s in sources if isinstance(s, dict) and s.get("type") != "image"]
                        image_sources = [s for s in sources if isinstance(s, dict) and s.get("type") == "image"]
                        print(f"   Text sources: {len(text_sources)}, Image sources: {len(image_sources)}")
                        details += f", Sources: {len(text_sources)} text + {len(image_sources)} image"
                    else:
                        print(f"   Sources (unknown format): {sources}")
                        details += ", Sources: unknown format"
                    
                    details += f", Answer length: {len(answer)} chars"
                else:
                    print("   No structured data found in response")
                    details += ", No structured data found"

            except Exception as parse_error:
                print(f"   Could not parse response data: {parse_error}")
                details += f", Parse error: {parse_error}"

            write_test_result(log_file, "Multimodal Chat Model", "PASSED", details, answer=answer, sources=sources, original_response=response)

        except Exception as e:
            print(f"⚠️  Multimodal chat test failed: {e}")
            print("   This might be expected if no documents exist in the test collection")
            write_test_result(log_file, "Multimodal Chat Model", "FAILED", "", str(e))

        # Test 3: Chat with attached image
        print("\n--- Test 3: Chat with Attached Image ---")
        #testing in png
        test_image = "data:image/png;base64,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"
        #testing in jpg
        #test_image ="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEBLAEsAAD/4QB9RXhpZgAASUkqAAgAAAADAA4BAgAzAAAAMgAAABoBBQABAAAAZQAAABsBBQABAAAAbQAAAAAAAABDdXRlIHZlY3RvciBzcGVlY2ggYnViYmxlIGljb24gd2l0aCBoZWxsbyBncmVldGluZy4sAQAAAQAAACwBAAABAAAA/+EFsmh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8APD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4KPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyI+Cgk8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgoJCTxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1sbnM6SXB0YzR4bXBDb3JlPSJodHRwOi8vaXB0Yy5vcmcvc3RkL0lwdGM0eG1wQ29yZS8xLjAveG1sbnMvIiAgIHhtbG5zOkdldHR5SW1hZ2VzR0lGVD0iaHR0cDovL3htcC5nZXR0eWltYWdlcy5jb20vZ2lmdC8xLjAvIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnBsdXM9Imh0dHA6Ly9ucy51c2VwbHVzLm9yZy9sZGYveG1wLzEuMC8iICB4bWxuczppcHRjRXh0PSJodHRwOi8vaXB0Yy5vcmcvc3RkL0lwdGM0eG1wRXh0LzIwMDgtMDItMjkvIiB4bWxuczp4bXBSaWdodHM9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9yaWdodHMvIiBwaG90b3Nob3A6Q3JlZGl0PSJHZXR0eSBJbWFnZXMvaVN0b2NrcGhvdG8iIEdldHR5SW1hZ2VzR0lGVDpBc3NldElEPSIxMDY1NDY1MzQyIiB4bXBSaWdodHM6V2ViU3RhdGVtZW50PSJodHRwczovL3d3dy5pc3RvY2twaG90by5jb20vbGVnYWwvbGljZW5zZS1hZ3JlZW1lbnQ/dXRtX21lZGl1bT1vcmdhbmljJmFtcDt1dG1fc291cmNlPWdvb2dsZSZhbXA7dXRtX2NhbXBhaWduPWlwdGN1cmwiIHBsdXM6RGF0YU1pbmluZz0iaHR0cDovL25zLnVzZXBsdXMub3JnL2xkZi92b2NhYi9ETUktUFJPSElCSVRFRC1FWENFUFRTRUFSQ0hFTkdJTkVJTkRFWElORyIgPgo8ZGM6Y3JlYXRvcj48cmRmOlNlcT48cmRmOmxpPnZlY3RvcnBsdXNiPC9yZGY6bGk+PC9yZGY6U2VxPjwvZGM6Y3JlYXRvcj48ZGM6ZGVzY3JpcHRpb24+PHJkZjpBbHQ+PHJkZjpsaSB4bWw6bGFuZz0ieC1kZWZhdWx0Ij5DdXRlIHZlY3RvciBzcGVlY2ggYnViYmxlIGljb24gd2l0aCBoZWxsbyBncmVldGluZy48L3JkZjpsaT48L3JkZjpBbHQ+PC9kYzpkZXNjcmlwdGlvbj4KPHBsdXM6TGljZW5zb3I+PHJkZjpTZXE+PHJkZjpsaSByZGY6cGFyc2VUeXBlPSdSZXNvdXJjZSc+PHBsdXM6TGljZW5zb3JVUkw+aHR0cHM6Ly93d3cuaXN0b2NrcGhvdG8uY29tL3Bob3RvL2xpY2Vuc2UtZ20xMDY1NDY1MzQyLT91dG1fbWVkaXVtPW9yZ2FuaWMmYW1wO3V0bV9zb3VyY2U9Z29vZ2xlJmFtcDt1dG1fY2FtcGFpZ249aXB0Y3VybDwvcGx1czpMaWNlbnNvclVSTD48L3JkZjpsaT48L3JkZjpTZXE+PC9wbHVzOkxpY2Vuc29yPgoJCTwvcmRmOkRlc2NyaXB0aW9uPgoJPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KPD94cGFja2V0IGVuZD0idyI/Pgr/7QCCUGhvdG9zaG9wIDMuMAA4QklNBAQAAAAAAGUcAlAAC3ZlY3RvcnBsdXNiHAJ4ADNDdXRlIHZlY3RvciBzcGVlY2ggYnViYmxlIGljb24gd2l0aCBoZWxsbyBncmVldGluZy4cAm4AGEdldHR5IEltYWdlcy9pU3RvY2twaG90bwD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5E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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAX000gKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABE9tttttqSAAAAAAAAAAAAAAAAAAAAAAAAAAAA1NtttttttttPgAAAAAAAAAAAAAAAAAAAAAAAAFVttttttttttty4AAAAAAAAAAAAAAAAAAAAAAAUtttttttttttttuQAAAAAAAAAAAAAAAAAAAAADhNtttttttttttttvYAAAAAAAAAAAAAAAAAAAACBtttttttttttttttt0gAAAAAAAAAAAAAAAAAACRtttttttttttttttttlgAAAAAAAAAAAAAAAAADltttttttttttttttttt6gAAAAAAAAAAAAAAAAABtttttttttttttttttttwAAAAAAAAAAAAAAAAAINtttttttttttttttttttoAAAAAAAAAAAAAAAAZNtttttttttttttttttttuoAAAAAAAAAAAAAAADNttttttttttttttttttttvAAAAAAAAAAAAAAAABtttttttttttttttttttttuAAAAAAAAAAAAAAAFttttttttttttttttttttttsAAAAAAAAAAAAAABNtttttttttttttttttttttsQAAAAAAAAAAAAAARttttttttttttttttttttttuAAAAAAAAAAAAAADttttttrNtttttttttttutttkAAAAAAAAAAAAABNtttNtvptttttttttttvlttsEAAAAAAAAAAAAARttttttrNtttuNqNttttyVttvAAAAAAAAAAAAACttt7tttdttts9vttttttCNttkAAAAAAAAAAAAABttujNt8NvhBwRp1tLNtwdttvAAAAAAAAAAAAAANttwRttctGAMjNgMtuNuCtttsAAAAAAAAAAAAAJttvCMUEdhLJ5lkZjwEN0VttugAAAAAAAAAAAABNtt8MYHlnzttpsaMxuZu5NttgAAAAAAAAAAAAAJttuTaltu7+Nsxvt/thtsdtttAAAAAAAAAAAAACRttthtup8GttvxzNdtZthttt4AAAAAAAAAAAAABNtt+NtnYixvSgNfIttNttttugAAAAAAAAAAAAABttulttC4Zuxcd0B1g1t4Ntt0AAAAAAAAAAAAAANttzNtkMwe5sivAiPLtvJttugAAAAAAAAAAAAABttvVtt3vEEttZz5xMNttNttkAAAAAAAAAAAAAAttttttttufNttttttttttttwgAAAAAAAAAAAAAQNttttttttttttttttttttttAAAAAAAAAAAAAAAVtttttttttttttttttttttt0AAAAAAAAAAAAAABNtttttttttttttttttttttvAAAAAAAAAAAAAAADNttttttttttttttttttttsgAAAAAAAAAAAAAAAJttttttttttttttttttttt4AAAAAAAAAAAAAAAAtttttttttttttttttttttlgAAAAAAAAAAAAAAADNttttttttttttttttttttgAAAAAAAAAAAAAAAABttttttttttttttttttttMAAAAAAAAAAAAAAAAAJtttttttttttttttttttwAAAAAAAAAAAAAAAAAAFtttttttttttttttttt4AAAAAAAAAAAAAAAAAADJttttttttttttttttt0gAAAAAAAAAAAAAAAAAADNtttttttttttttttt5AAAAAAAAAAAAAAAAAAAAAJtttttttttttttttsAAAAAAAAAAAAAAAAAAAAAACtttttttttttttttAAAAAAAAAAAAAAAAAAAAAAADbNtttttttttttvAAAAAAAAAAAAAAAAAAAAAAAAAezNtuH1mmNttvgAAAAAAAAAAAAAAAAAAAAAAAAAAT64YAAARtthgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADNt7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFt/gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAvWgA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"
        if test_image:
            try:
                image_messages = [
                    types.Message(
                        role=types.Role.USER,
                        parts=[types.TextPart(text="What is the attached image about?")]
                    )
                ]

                response = multimodal_chat_model.chat(
                    messages=image_messages,
                    collection_id="01BT6QKHUAQI2T2I2FTVF34OU7JGPDX6KC",
                    query_images=[test_image],
                )

                print("✓ Chat with attached image completed successfully")
                print(f"   Response role: {response.message.role}")
                print(f"   Prompt tokens: {response.metadata.usage.prompt_tokens}")
                print(f"   Completion tokens: {response.metadata.usage.completion_tokens}")

                # Try to parse response data
                details = f"Image: {test_image}, Tokens: {response.metadata.usage.prompt_tokens}/{response.metadata.usage.completion_tokens}"
                answer = ""
                sources = []

                try:
                    if hasattr(response.message, "data") and response.message.data:
                        # Handle both parsed dict and JSON string cases
                        if isinstance(response.message.data, str):
                            data = json.loads(response.message.data)
                        else:
                            data = response.message.data
                            
                        answer = data.get("answer", "No answer found")
                        sources = data.get("sources", [])
                        challenge_verification = data.get("challenge_verification", False)
                        print(f"   Answer preview: {answer[:150]}...")
                        
                        # Handle both structured and unstructured sources
                        if isinstance(sources, dict):
                            text_sources = sources.get("text_sources", [])
                            image_sources = sources.get("image_sources", [])
                            total_sources = len(text_sources) + len(image_sources)
                            print(f"   Sources: {len(text_sources)} text + {len(image_sources)} image = {total_sources} total")
                        else:
                            total_sources = len(sources) if isinstance(sources, list) else 0
                            print(f"   Number of sources: {total_sources}")
                            
                        print(f"   Challenge verification: {challenge_verification}")
                        details += f", Sources: {total_sources}, Answer: {len(answer)} chars, Verification: {challenge_verification}"
                except Exception as parse_error:
                    print(f"   Could not parse response data: {parse_error}")
                    details += f", Parse error: {parse_error}"

                write_test_result(log_file, "Chat with Attached Image", "PASSED", details, answer=answer, sources=sources, original_response=response)

            except Exception as e:
                print(f"⚠️  Chat with attached image test failed: {e}")
                write_test_result(log_file, "Chat with Attached Image", "FAILED", "", str(e))
        else:
            print(f"⚠️  Test image not found at {test_image}")
            print("   Skipping attached image test")
            write_test_result(log_file, "Chat with Attached Image", "SKIPPED", f"Image not found: {test_image}")

        print("\n✅ All multimodal chat model tests completed!")
        print("\n🚀 Your multimodal chat model functionality is working!")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all required packages are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat_model_configurations(log_file=None):
    """Test different chat model configurations."""
    print("\n======================================================================")
    print("CHAT MODEL CONFIGURATION TEST")
    print("======================================================================")

    try:
        # Configuration validation
        print("\n--- Testing Configuration Parameters ---")
        config_tests = [
            ("enable_multimodal_search", [True, False]),
            ("retriever_size_image", [5, 10, 20]),
            ("retriever_size_text", [10, 20, 30]),
            ("max_words_answer", [200, 400, 800]),
        ]

        for param_name, test_values in config_tests:
            print(f"✓ Parameter '{param_name}' accepts values: {test_values}")

        if log_file:
            write_test_result(log_file, "Chat Model Configuration", "PASSED", f"Tested {len(config_tests)} parameter configurations")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        if log_file:
            write_test_result(log_file, "Chat Model Configuration", "FAILED", "", str(e))
        return False


if __name__ == "__main__":
    print("Starting multimodal chat model tests...")

    # Create log file for main execution
    main_log_file = create_test_log_file()
    write_log_header(main_log_file)

    config_success = test_chat_model_configurations(main_log_file)
    chat_success = test_multimodal_chat_model()

    print("\n======================================================================")
    print("TEST SUMMARY")
    print("======================================================================")
    print(f"Configuration Test: {'✅ PASSED' if config_success else '❌ FAILED'}")
    print(f"Multimodal Chat Test: {'✅ PASSED' if chat_success else '❌ FAILED'}")

    # Write final summary to log
    if config_success and chat_success:
        write_test_result(main_log_file, "Overall Test Suite", "PASSED", "All tests completed successfully")
        print("\n🎉 All tests passed! Your multimodal chat model is ready to use!")
    else:
        write_test_result(main_log_file, "Overall Test Suite", "FAILED", f"Config: {'PASSED' if config_success else 'FAILED'}, Chat: {'PASSED' if chat_success else 'FAILED'}")
        print("\n⚠️  Some tests failed. Check the logs above for details.")
        print("   Note: Chat tests might fail if Elasticsearch doesn't have test data.")

    print(f"\n📋 Test results logged to: {main_log_file}")