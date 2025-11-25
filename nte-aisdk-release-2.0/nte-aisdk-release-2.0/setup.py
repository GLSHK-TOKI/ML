from setuptools import setup

setup(
    name="nte_aisdk",
    version="2.0.0a2",
    description="The AI Python library provided by NTE",
    author="nte",
    author_email="Example@cathaypacific.com",
    readme="README.md",
    install_requires=[
        "langchain",
        "elasticsearch",
        "langchain-openai>=0.3.8",
        "flask",
        "azure-identity",
        "pytz",
        "python-jose",
        "croniter",
        "google-genai>=1.16.1",
    ],
    package_data={
        "nte_aisdk": ["py.typed"],
    },
)
