"""
Pipeline: Step 00 - Authentication
Generates and secures access tokens for the Fyers API.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.auth import FyersAuth  # noqa: E402
from loguru import logger  # noqa: E402


def main():
    try:
        auth = FyersAuth()
        session = auth.get_session()
        auth.generate_auth_url(session)

        print()
        print("INSTRUCTIONS:")
        print("1. Log in to your Fyers account in the opened browser")
        print("2. Authorize the application")
        print("3. Copy the 'auth_code' from the redirected URL")

        auth_code = input("\nEnter auth_code: ").strip()
        if not auth_code:
            logger.error("No auth code provided")
            return 1

        auth.get_access_token(session, auth_code)
        logger.info("Access token successfully generated and saved.")
        return 0
    except Exception as e:
        logger.exception(f"Authentication failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
