"""
Script 00: Generate Fyers Access Token

Generates Fyers API access token for authentication.
This needs to be run ONCE before downloading data.

Usage:
    python scripts/00_generate_token.py

Output:
    - Prints access token to console
    - Update .env file with the token
"""

import os
import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

# Load environment
load_dotenv()

# =====================
# CONFIGURATION
# =====================

CLIENT_ID = os.getenv("CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
REDIRECT_URI = "https://127.0.0.1/"

# Validate credentials
if not CLIENT_ID or not SECRET_KEY:
    print("❌ ERROR: CLIENT_ID and SECRET_KEY not found in .env file")
    print("\nPlease create a .env file with:")
    print("  CLIENT_ID=your_client_id")
    print("  SECRET_KEY=your_secret_key")
    sys.exit(1)

print("="*60)
print("FYERS ACCESS TOKEN GENERATION")
print("="*60)
print(f"Client ID: {CLIENT_ID}")
print(f"Redirect URI: {REDIRECT_URI}")
print("="*60)

# =====================
# STEP 1: Create Session
# =====================

print("\nStep 1: Creating Fyers session...")

try:
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )
    print("✅ Session created successfully")
except Exception as e:
    print(f"❌ Failed to create session: {e}")
    sys.exit(1)

# =====================
# STEP 2: Generate Auth URL
# =====================

print("\nStep 2: Generating authorization URL...")

try:
    auth_url = session.generate_authcode()
    print("✅ Authorization URL generated")
    print(f"\n{auth_url}\n")
except Exception as e:
    print(f"❌ Failed to generate auth URL: {e}")
    sys.exit(1)

# =====================
# STEP 3: Open Browser
# =====================

print("Step 3: Opening browser for authentication...")
print("(If browser doesn't open, copy the URL above manually)")

try:
    webbrowser.open(auth_url)
    print("✅ Browser opened")
except:
    print("⚠️  Could not open browser automatically")

print("\n" + "="*60)
print("INSTRUCTIONS:")
print("="*60)
print("1. Log in to your Fyers account in the opened browser")
print("2. Authorize the application")
print("3. You will be redirected to https://127.0.0.1/?auth_code=...")
print("4. Copy the 'auth_code' from the URL")
print("5. Paste it below")
print("="*60)

# =====================
# STEP 4: Get Auth Code
# =====================

auth_code = input("\nEnter auth_code: ").strip()

if not auth_code:
    print("❌ No auth code provided")
    sys.exit(1)

print(f"\n✅ Auth code received: {auth_code[:20]}...")

# =====================
# STEP 5: Generate Token
# =====================

print("\nStep 5: Generating access token...")

try:
    session.set_token(auth_code)
    response = session.generate_token()
    
    if response.get("s") == "ok":
        access_token = response.get("access_token")
        
        print("\n" + "="*60)
        print("✅ ACCESS TOKEN GENERATED SUCCESSFULLY")
        print("="*60)
        print(f"\nAccess Token:\n{access_token}")
        print("\n" + "="*60)
        
        # Save to .env
        env_path = Path(".env")
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            # Check if ACCESS_TOKEN already exists
            if "ACCESS_TOKEN=" in env_content:
                # Replace existing token
                lines = env_content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith("ACCESS_TOKEN="):
                        new_lines.append(f"ACCESS_TOKEN={access_token}")
                    else:
                        new_lines.append(line)
                env_content = '\n'.join(new_lines)
            else:
                # Append new token
                env_content += f"\nACCESS_TOKEN={access_token}\n"
            
            with open(env_path, 'w') as f:
                f.write(env_content)
            
            print("✅ Access token saved to .env file")
        else:
            print("\n⚠️  .env file not found")
            print("Please manually add this line to your .env file:")
            print(f"ACCESS_TOKEN={access_token}")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Verify .env file contains ACCESS_TOKEN")
        print("2. Run: python scripts/01_download_nifty50.py")
        print("="*60)
        
    else:
        print(f"\n❌ Token generation failed: {response}")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error generating token: {e}")
    sys.exit(1)
