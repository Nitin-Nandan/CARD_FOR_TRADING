"""
Fyers Authentication Utility
"""

import os
import webbrowser
from pathlib import Path
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
from loguru import logger


class FyersAuth:
    """
    Manages Fyers API authentication and token lifecycle.
    """

    def __init__(
        self, client_id=None, secret_key=None, redirect_uri="https://127.0.0.1/"
    ):
        load_dotenv()
        self.client_id = client_id or os.getenv("CLIENT_ID")
        self.secret_key = secret_key or os.getenv("SECRET_KEY")
        self.redirect_uri = redirect_uri

        if not self.client_id or not self.secret_key:
            raise ValueError("CLIENT_ID and SECRET_KEY must be in .env or provided.")

    def get_session(self):
        """Initialize session model"""
        return fyersModel.SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )

    def generate_auth_url(self, session):
        """Step 1 & 2: Generate URL and open browser"""
        url = session.generate_authcode()
        logger.info(f"Auth URL: {url}")
        try:
            webbrowser.open(url)
        except Exception:
            logger.warning("Could not open browser. Please use the URL above manually.")
        return url

    def get_access_token(self, session, auth_code):
        """Step 3: Exchange code for token"""
        session.set_token(auth_code)
        response = session.generate_token()

        if response.get("s") == "ok":
            token = response.get("access_token")
            self._save_to_env(token)
            return token
        else:
            raise RuntimeError(f"Token generation failed: {response}")

    def _save_to_env(self, token):
        """Update .env file with new token"""
        path = Path(".env")
        new_content = f"ACCESS_TOKEN={token}\n"

        if path.exists():
            lines = path.read_text().splitlines()
            others = [line for line in lines if not line.startswith("ACCESS_TOKEN=")]
            new_content = "\n".join(others + [f"ACCESS_TOKEN={token}"])

        path.write_text(new_content)
        logger.info("Saved ACCESS_TOKEN to .env")
