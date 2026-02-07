import os

from dotenv import load_dotenv

import wandb


def setup_wandb():
    """Load environment variables and login to WandB."""
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    elif os.getenv("WANDB_MODE") in ["disabled", "offline"]:
        pass
    else:
        # Only prompt/login if not explicitly disabled and no unexpected CI environment
        wandb.login()
