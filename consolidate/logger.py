import logging
from datetime import datetime

# Configure logging
log_filename = f"logs_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger()
