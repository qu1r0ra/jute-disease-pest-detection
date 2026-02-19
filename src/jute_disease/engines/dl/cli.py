from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI


def main() -> None:
    """Entry point for the unified Deep Learning CLI."""
    load_dotenv()
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
