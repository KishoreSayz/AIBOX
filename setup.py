from setuptools import setup, find_packages

setup(
    name="llmkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
        "httpx>=0.24.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.5.0"],
        "google": ["google-generativeai>=0.3.0"],
        "huggingface": ["huggingface-hub>=0.16.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "google-generativeai>=0.3.0",
            "huggingface-hub>=0.16.0",
        ],
    },
    python_requires=">=3.9",
) 