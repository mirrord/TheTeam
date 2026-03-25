"""Utility functions for pithos."""

import logging
import requests
import json
import os

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_available_models(ollama_url=OLLAMA_BASE_URL):
    """
    Retrieves a list of installed Ollama models from the local Ollama server.

    Args:
        ollama_url (str): The URL of the Ollama server. Defaults to the configured OLLAMA_BASE_URL.

    Returns:
        list: A list of model names (strings) installed in Ollama, or an empty list if an error occurs.
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        model_names = [model["model"] for model in data.get("models", [])]
        return model_names
    except requests.exceptions.RequestException as e:
        logger.error("Error connecting to Ollama server or retrieving models: %s", e)
        return []
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON response from Ollama server: %s", e)
        return []
