from pathlib import Path

import yaml
from dotenv import load_dotenv


# Determine the root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[3]    #__file__ is the path to the python file that is currently being excuted
                                                  #.partens[3] goes three levels up in the path

# Load local environment variables from the .env file
load_dotenv(ROOT_DIR / ".env")


def load_settings() -> dict:
    """Load project settings from config/settings.yaml."""

    config_path = ROOT_DIR / "config" / "settings.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)           # transforms yaml into dictionary structure

    return settings                               #returns created dictionary
 

# Load the settings once when this module is imported / Load and store the project settings
SETTINGS = load_settings()
