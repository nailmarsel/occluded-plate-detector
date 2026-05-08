import logging
from app.core.config import settings


def setup_logging() -> logging.Logger:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("autobahncv")


logger = setup_logging()
