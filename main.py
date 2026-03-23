import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# ---------- Load environment variables ----------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# ---------- Gemini setup ----------
genai.configure(api_key=GEMINI_API_KEY)
# Use a model that exists – check with genai.list_models() if needed
model = genai.GenerativeModel('gemini-2.5-flash')

# ---------- Google OAuth scopes ----------
SCOPES = [
    'https://www.googleapis.com/auth/tasks',
    'https://www.googleapis.com/auth/calendar.events'
]
TOKEN_URI = 'https://oauth2.googleapis.com/token'

def get_google_credentials():
    """Obtain fresh credentials using the stored refresh token."""
    creds = Credentials(
        token=None,
        refresh_token=REFRESH_TOKEN,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_uri=TOKEN_URI,
        scopes=SCOPES
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

# ---------- FastAPI app ----------
app = FastAPI()

# CORS – allow your frontend domains
origins = [
    "https://kalanasumanaweera.github.io",
    "http://localhost:8000",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceText(BaseModel):
    text: str

@app.post("/process_task")
async def process_task(item: VoiceText):
    try:
        # Current date/time in Sri Lanka (UTC+5:30)
        sri_lanka_tz = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(sri_lanka_tz)
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # ----- 1. Gemini prompt -----
        prompt = f"""
Today is {current_date} in Sri Lanka (UTC+5:30). The current time is {current_time}.

The following text is in Sinhala or English. Extract details and output a JSON object.

Rules:
- content: the event/task title (keep in original language).
- start_datetime: an ISO 8601 datetime string in UTC with time (e.g., "2026-03-24T03:30:00Z") if a specific time is mentioned; otherwise null.
- date: an ISO 8601 date string (YYYY-MM-DD) if only a date is mentioned but no time; otherwise null.
- description: any extra notes (original language), otherwise null.
- labels: list of labels (optional, null if none).

Date/Time Interpretation:
- Use today's date ({current_date}) as the reference point.
- All times are in Sri Lanka time (UTC+5:30). Convert them to UTC for start_datetime.
- If a time is mentioned (e.g., "at 9am"), use start_datetime with the full datetime.
- If only a date is mentioned (e.g., "tomorrow"), use date with that date, and leave start_datetime null.

Examples:
* "I have a meeting tomorrow at 9.00 am" → content: "Meeting", start_datetime: "2026-03-24T03:30:00Z", date: null, description: "I have a meeting"
* "Buy milk tomorrow" → content: "Buy milk", start_datetime: null, date: "2026-03-24", description: null
* "Call John" → content: "Call John", start_datetime: null, date: null, description: null

Return only the JSON object, no other text.

Text: {item.text}
"""
        response = model.generate_content(prompt)
        task_data_str = response.text.strip()
        if task_data_str.startswith("```json"):
            task_data_str = task_data_str[7:-3].strip()
        task_json = json.loads(task_data_str)
        print("Gemini output:", json.dumps(task_json, indent=2))

        # ----- 2. Get credentials and service -----
        creds = get_google_credentials()

        # ----- 3. Decide: calendar event (with time) or fallback -----
        if task_json.get('start_datetime'):
            # Create a Google Calendar event
            calendar_service = build('calendar', 'v3', credentials=creds)
            event = {
                'summary': task_json['content'],
                'description': task_json.get('description', ''),
                'start': {
                    'dateTime': task_json['start_datetime'],
                    'timeZone': 'UTC',   # already UTC
                },
                'end': {
                    'dateTime': task_json['start_datetime'],   # same time for a task‑like event
                    'timeZone': 'UTC',
                },
                'reminders': {
                    'useDefault': True,   # use your default calendar reminders
                },
            }
            # Append labels to description if any
            if task_json.get('labels'):
                event['description'] += f"\n\nLabels: {', '.join(task_json['labels'])}"
            created_event = calendar_service.events().insert(calendarId='primary', body=event).execute()
            print("Calendar event created:", json.dumps(created_event, indent=2))
            return {
                "status": "success",
                "event": {
                    "id": created_event['id'],
                    "title": created_event['summary'],
                    "link": created_event.get('htmlLink')
                }
            }

        else:
            # Fallback to Google Tasks (date‑only or no date)
            tasks_service = build('tasks', 'v1', credentials=creds)
            # Get default task list
            task_lists = tasks_service.tasklists().list().execute()
            if not task_lists.get('items'):
                default_list = tasks_service.tasklists().insert(body={'title': 'My Tasks'}).execute()
                task_list_id = default_list['id']
            else:
                task_list_id = task_lists['items'][0]['id']

            task_body = {
                'title': task_json['content'],
                'notes': task_json.get('description', '')
            }
            if task_json.get('date'):
                task_body['due'] = task_json['date']   # YYYY-MM-DD
            # Append labels to notes
            if task_json.get('labels'):
                labels_text = f"\n\nLabels: {', '.join(task_json['labels'])}"
                task_body['notes'] = (task_body.get('notes', '') + labels_text).strip()

            created_task = tasks_service.tasks().insert(tasklist=task_list_id, body=task_body).execute()
            return {
                "status": "success",
                "task": {
                    "id": created_task['id'],
                    "title": created_task['title']
                }
            }

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok"}