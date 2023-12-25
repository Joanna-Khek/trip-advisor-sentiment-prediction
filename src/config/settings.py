from pathlib import Path
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config" / "main.yaml"
DATASET_DIR = ROOT / "data"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"
TRAINED_MODEL_DIR = ROOT / "models"

def fetch_config_from_yaml() -> yaml:
    """Parse YAML containing the package configuration"""

    with open(CONFIG_FILE_PATH, 'r') as yaml_file:
        parsed_config = yaml.safe_load(yaml_file)
        return parsed_config

config = fetch_config_from_yaml()