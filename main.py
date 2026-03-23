import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')  # Make sure this model exists (or use gemini-2.0-flash-exp)

# Google Tasks OAuth scopes
SCOPES = ['https://www.googleapis.com/auth/tasks']
TOKEN_URI = 'https://oauth2.googleapis.com/token'

def get_google_credentials():
    """Use the refresh token to obtain fresh credentials."""
    creds = Credentials(
        token=None,
        refresh_token=REFRESH_TOKEN,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_uri=TOKEN_URI,
        scopes=SCOPES
    )
    # If token is expired, refresh it
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

# FastAPI app
app = FastAPI()

origins = [
    "https://kalanasumanaweera.github.io",  # GitHub Pages domain
    # "https://your-netlify-app.netlify.app",  # if using Netlify
    "http://localhost:8000",  # for local testing
    "http://localhost:5500",  # if using VS Code Live Server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for development, but restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceText(BaseModel):
    text: str

@app.post("/process_task")
async def process_task(item: VoiceText):
    try:
        # 1. Use Gemini to extract task details
        prompt = f"""
The following text is in Sinhala (සිංහල) or English. Extract task details from it.
Return a JSON object with these fields:
- content: the task title (keep in original language)
- due: an ISO 8601 date string (YYYY-MM-DD) or date-time string (YYYY-MM-DDTHH:MM:SSZ) if a date/time is mentioned; otherwise null
- description: any extra notes (in original language), otherwise null
- labels: list of labels/projects (optional, in original language)

For due:
- If only a date is mentioned, use "YYYY-MM-DD" (e.g., tomorrow → "2025-03-24").
- If a time is mentioned, use "YYYY-MM-DDTHH:MM:SSZ" (UTC). Assume time in Sri Lanka time (UTC+5:30) and convert to UTC.
- If no date, set due to null.

Text: {item.text}
"""
        response = model.generate_content(prompt)
        task_data_str = response.text.strip()
        
        # Remove markdown code fences if present
        if task_data_str.startswith("```json"):
            task_data_str = task_data_str[7:-3].strip()
        
        task_json = json.loads(task_data_str)
        
        # 2. Get Google Tasks credentials
        creds = get_google_credentials()
        service = build('tasks', 'v1', credentials=creds)
        
        # 3. Get the default task list
        task_lists = service.tasklists().list().execute()
        if not task_lists.get('items'):
            # Create a default task list if none exists
            default_list = service.tasklists().insert(body={'title': 'My Tasks'}).execute()
            task_list_id = default_list['id']
        else:
            task_list_id = task_lists['items'][0]['id']
        
        # 4. Build the task object
        task_body = {
            'title': task_json['content'],
            'notes': task_json.get('description', '')
        }
        if task_json.get('due'):
            task_body['due'] = task_json['due']   # now in ISO format

        # Ensure 'due' is in full RFC 3339 format (Google Tasks requires time component)
        if 'due' in task_body and 'T' not in task_body['due']:
            task_body['due'] += 'T00:00:00Z'

        # Append labels to notes if present
        if task_json.get('labels'):
            labels_text = f"\n\nLabels: {', '.join(task_json['labels'])}"
            task_body['notes'] = (task_body.get('notes', '') + labels_text).strip()
        
        # 5. Insert task
        created_task = service.tasks().insert(tasklist=task_list_id, body=task_body).execute()
        
        return {
            "status": "success",
            "task": {
                "id": created_task['id'],
                "title": created_task['title']
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok"}