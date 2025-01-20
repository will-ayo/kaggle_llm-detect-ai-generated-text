import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

fastapi_options = {
    "title": config.get("API").get("NAME"),
    "version": "0.1"
}

mlflow_options = {
    "is_active": config.get("ML_FLOW").get("IS_ACTIVE", False),
    "uri": config.get("ML_FLOW").get("URI"),
    "name": config.get("ML_FLOW").get("NAME"),
}
