import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/tasks']

def main():
    flow = InstalledAppFlow.from_client_secrets_file(
        'client_secret.json', SCOPES)
    # Run local server to get authorization
    creds = flow.run_local_server(port=0)
    
    # Print refresh token
    print("Refresh token:", creds.refresh_token)
    # Also save full credentials to token.json for local testing
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

if __name__ == '__main__':
    main()