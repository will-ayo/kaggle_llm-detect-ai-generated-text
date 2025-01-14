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
    "title": config["API"]["NAME"],
    "version": "0.1"
}
