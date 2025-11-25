import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from nte_aisdk.pii_detection import PIIDetector
from nte_aisdk.providers.azure import AzureProvider
from nte_aisdk.types import PIICategory, PIIDetectionConfig
load_dotenv()
jsonl_file_path = "<YOUR_TEST_JSONL_FILE_PATH>"

# Initialize Azure provider
azure_provider = AzureProvider(
    azure_endpoint=os.getenv("<AZURE_ENDPOINT>"),
    api_key=os.getenv("<AZURE_API_KEY>"),
    api_version=os.getenv("<AZURE_API_VERSION>"),
)

# Create language model
language_model = azure_provider.create_language_model(
    azure_deployment="<YOUR_DEPLOYMENT_NAME>",
    model_name="<YOUR_MODEL_NAME>",
)

# Configure PII categories
pii_config = PIIDetectionConfig(
    categories=[
       PIICategory(
            name="Email Address",
            definition="Email addresses in standard format (e.g., user@example.com)"
        ),
        PIICategory(
            name="Phone Number",
            definition="Phone numbers in various formats including international, national, and local formats"
        ),
        PIICategory(
            name="Credit Card Number",
            definition="Credit card numbers with 13-19 digits, may include spaces or dashes"
        ),
        PIICategory(
            name="Social Security Number",
            definition="US Social Security Numbers in format XXX-XX-XXXX or XXXXXXXXX"
        ),
        PIICategory(
            name="Personal Name",
            definition="Full names of individuals including first and last names"
        ),
        PIICategory(
            name="Address",
            definition="Physical addresses including street addresses, postal codes, and locations"
        ),
        PIICategory(
            name="Date of Birth",
            definition="Birth dates in various formats (MM/DD/YYYY, DD/MM/YYYY, etc.)"
        ),
        PIICategory(
            name="Bank Account Number",
            definition="Bank account numbers and routing numbers"
        ),
    ]
)

# Initialize PII detector
pii_detector = PIIDetector(
    language_model=language_model,
    config=pii_config
)

def wait_for_batch_completion(batch_id: str, description: str = "batch", max_wait: int = 1800, check_interval: int = 10, **retrieve_kwargs):
    start_time = time.time()
    while time.time() - start_time < max_wait:
        batch_response = pii_detector.batches.retrieve(batch_id=batch_id, **retrieve_kwargs)

        if isinstance(batch_response, dict):
            status = batch_response.get("status", "unknown")
            if status in ["failed", "expired", "cancelled"]:
                error_msg = f"Batch failed: {batch_response.get('error', status)}"
                raise RuntimeError(error_msg)
            print(f"{description} status: {status}. Waiting...")
            time.sleep(check_interval)
            continue

        # If response is a Pydantic model, batch is completed
        if hasattr(batch_response, "status") and batch_response.status == "completed":
            return batch_response

        # Unexpected response type
        print(f"Unexpected response type: {type(batch_response)}. Waiting...")
        time.sleep(check_interval)

    error_msg = f"{description.capitalize()} timeout after {max_wait}s"
    raise TimeoutError(error_msg)


def test_pii_batch():
    # Use the existing JSONL file
    try:
        # Step 1: Submit PII detection batch directly from JSONL file
        print(f"\n1. Submitting PII detection batch from JSONL file: {jsonl_file_path}")
        input_texts = []
        with Path(jsonl_file_path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.strip())
                    text = data.get("text")
                    if text:
                        input_texts.append(text)
                except json.JSONDecodeError:
                    continue

        markdown_batch_id = pii_detector.batches.submit_detect(
            input_texts=input_texts
        )
        print(f"✅ Markdown batch submitted: {markdown_batch_id}")


        # Step 2: Wait for completion and get results (with manual polling)
        print("\n2. Polling for markdown batch completion...")
        markdown_results = wait_for_batch_completion(markdown_batch_id, "markdown batch")

        # Save markdown results to file without any alterations
        markdown_results_file = Path("./results/markdown_batch_results.json")
        markdown_results_file.parent.mkdir(parents=True, exist_ok=True)
        with markdown_results_file.open("w", encoding="utf-8") as f:
            json.dump(markdown_results.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"✅ Markdown results : {markdown_results_file}")


        # Step 3: Submit JSON conversion batch
        print("\n3. Submitting JSON conversion batch...")
        json_batch_id = pii_detector.batches.submit_detect_to_json(
            markdown_results=markdown_results
        )
        print(f"✅ JSON batch submitted: {json_batch_id}")

        # Step 4: Wait for completion and get automatically processed results
        print("\n4. Polling for JSON batch completion and auto-processing...")
        detection_results = wait_for_batch_completion(
            json_batch_id, 
            "JSON batch",
            original_texts=input_texts,
            enable_location_mark=True
        )

        # Save structured results to JSON (with proper serialization)
        detection_results_file = Path("./results/detect_to_json_results.json")
        detection_results_file.parent.mkdir(parents=True, exist_ok=True)
        with detection_results_file.open("w", encoding="utf-8") as f:
            # Convert Pydantic model to dict for JSON serialization
            json.dump(detection_results.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"\n✅ Structured detection results saved to: {detection_results_file}")

        # Step 5: Submit masking batch using structured detection response and original JSONL
        print("\n5. Submitting PII masking batch...")
        mask_batch_id = pii_detector.batches.submit_mask(
            detection_response=detection_results,  # Use structured detection response
            input_texts=input_texts   # Original input JSONL input texts
        )

        # Step 6: Wait for masking completion and get structured results
        print("\n6. Polling for masking completion and auto-processing...")
        masking_results = wait_for_batch_completion(mask_batch_id, "masking batch")

        # Step 7: Save structured masking results to JSON
        print("\n7. Saving structured masking results...")
        masking_results_file = Path("./results/structured_masking_results.json")
        masking_results_file.parent.mkdir(parents=True, exist_ok=True)
        with masking_results_file.open("w", encoding="utf-8") as f:
            # Convert Pydantic model to dict for JSON serialization
            json.dump(masking_results.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"✅ Structured masking results saved to: {masking_results_file}")
        print(f"\n✅ SUCCESS: Complete structured batch PII detection and masking pipeline completed!")

    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError, KeyError) as e:
        print(f"❌ Error: {e}")


test_pii_batch()
