import logging

APP_NAME = "llm-api"

LOGGER = logging.getLogger(APP_NAME)
LOGGER.setLevel(logging.INFO)

fastapi_options = {
    "title": APP_NAME,
    "version": "0.1"
}