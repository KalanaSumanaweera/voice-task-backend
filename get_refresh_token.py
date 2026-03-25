import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    'https://www.googleapis.com/auth/tasks',
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/calendar.readonly'   # required for freebusy query
]

def main():
    flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
    creds = flow.run_local_server(port=0)
    print("Refresh token:", creds.refresh_token)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

if __name__ == '__main__':
    main()