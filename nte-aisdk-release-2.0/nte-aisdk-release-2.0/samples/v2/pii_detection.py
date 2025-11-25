import os

from dotenv import load_dotenv

from nte_aisdk.pii_detection import PIIDetector
from nte_aisdk.providers.azure import AzureProvider
from nte_aisdk.types import PIICategory, PIIDetectionConfig

load_dotenv()

# Initialize Azure provider
azure_provider = AzureProvider(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

# Create language model for PII detection
language_model = azure_provider.create_language_model(
    azure_deployment="gpt-4.1-mini",
    model_name="gpt-4.1-mini",
)

# Configure PII categories to detect
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

def test_pii_detection():
    """Test PII detection functionality."""
    print("Testing PII Detection:")
    test_text = "Contact me at test@example.com or call 555-0123"

    result_no_location = pii_detector.detect(test_text, enable_location_mark=False)
    print(f"Input: {test_text}")
    print(f"Result without location: {result_no_location.data}")

    result_with_location = pii_detector.detect(test_text, enable_location_mark=True)
    print(f"Result with location: {result_with_location.data}")
    print(f"Usage: {result_with_location.usage}")
    print(f"Confidence: {result_with_location.confidence}")

def test_pii_masking():
    """Test PII masking functionality."""
    print("Testing PII Masking:")
    test_text = "My email is john.doe@example.com and phone is 555-123-4567"

    # First detect PII
    detection_result = pii_detector.detect(test_text, enable_location_mark=False)
    print(f"Original text: {test_text}")
    print(f"Detected PII: {detection_result.data}")

    # Then mask the PII
    if detection_result.data:
        mask_result = pii_detector.mask(test_text, detection_result.data)
        print(f"Masked text: {mask_result.text}")
        print(f"Token usage: {mask_result.usage}")
    else:
        print("No PII detected to mask")

# Run the tests
test_pii_detection()
test_pii_masking()
