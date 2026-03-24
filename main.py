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

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')  # or gemini-1.5-flash

SCOPES = [
    'https://www.googleapis.com/auth/tasks',
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/calendar.readonly'   # needed for freebusy query
]
TOKEN_URI = 'https://oauth2.googleapis.com/token'

def get_google_credentials():
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

app = FastAPI()
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

def get_free_slots(calendar_service, date_str, duration_minutes=60):
    """Return list of (start, end) UTC datetime intervals free for at least duration_minutes."""
    start = datetime.fromisoformat(f"{date_str}T00:00:00+00:00")
    end = datetime.fromisoformat(f"{date_str}T23:59:59+00:00")
    body = {
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "items": [{"id": "primary"}]
    }
    freebusy = calendar_service.freebusy().query(body=body).execute()
    busy = [(datetime.fromisoformat(b['start']), datetime.fromisoformat(b['end']))
            for b in freebusy['calendars']['primary']['busy']]
    busy.sort()
    merged = []
    for b in busy:
        if not merged or b[0] > merged[-1][1]:
            merged.append(b)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b[1]))
    free_slots = []
    current = start
    for b_start, b_end in merged:
        if b_start > current:
            if (b_start - current).total_seconds() >= duration_minutes * 60:
                free_slots.append((current, b_start))
        current = b_end
    if current < end:
        free_slots.append((current, end))
    return free_slots

@app.post("/process_task")
async def process_task(item: VoiceText):
    try:
        sri_lanka_tz = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(sri_lanka_tz)
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        prompt = f"""
Today is {current_date} in Sri Lanka (UTC+5:30). The current time is {current_time}.

The following text is in Sinhala or English. Extract details and output a JSON object.

Rules:
- content: the event/task title (keep in original language).
- intent: one of "schedule" (specific time given), "find_slot" (user asks for a suitable time), or "task" (simple task).
- date: ISO date (YYYY-MM-DD) if a date is mentioned; otherwise null.
- start_datetime: full ISO datetime (with Z) if a specific time is given; otherwise null.
- duration_minutes: estimated duration (default 60) if intent is "find_slot".
- preferred_time: "morning", "afternoon", "evening", or null.
- description: any extra notes, otherwise null.
- labels: list of labels, null if none.

Date/Time Interpretation:
- Use today's date ({current_date}) as reference.
- All times are Sri Lanka time, convert to UTC for start_datetime.
- For "find_slot", we will automatically pick a free time on the given date.

Examples:
* "I have a meeting tomorrow at 9.00 am" → intent: "schedule", start_datetime: "2026-03-24T03:30:00Z", date: null
* "හෙට සුදුසු වෙලාවක් බලලා මට ගෙදර වැඩ කරන්න දාන්න" → content: "වෙලාවක් බලලා", intent: "find_slot", date: "2026-03-24", duration_minutes: 60, preferred_time: null
* "Buy milk tomorrow" → intent: "task", date: "2026-03-24"
* "Call John" → intent: "task", date: null

Return only the JSON object, no other text.

Text: {item.text}
"""
        response = model.generate_content(prompt)
        task_data_str = response.text.strip()
        if task_data_str.startswith("```json"):
            task_data_str = task_data_str[7:-3].strip()
        task_json = json.loads(task_data_str)
        print("Gemini output:", json.dumps(task_json, indent=2))

        creds = get_google_credentials()
        intent = task_json.get('intent', 'task')

        # ----- Handle "find_slot" -----
        if intent == 'find_slot':
            date_to_check = task_json.get('date')
            if not date_to_check:
                tomorrow = datetime.now(sri_lanka_tz) + timedelta(days=1)
                date_to_check = tomorrow.strftime("%Y-%m-%d")
            duration = task_json.get('duration_minutes', 60)
            calendar_service = build('calendar', 'v3', credentials=creds)
            free_slots = get_free_slots(calendar_service, date_to_check, duration)
            if free_slots:
                chosen_start, chosen_end = free_slots[0]   # simplest: first slot
                start_iso = chosen_start.isoformat() + 'Z'
                end_iso = chosen_end.isoformat() + 'Z'
                event = {
                    'summary': task_json['content'],
                    'description': task_json.get('description', ''),
                    'start': {'dateTime': start_iso, 'timeZone': 'UTC'},
                    'end': {'dateTime': end_iso, 'timeZone': 'UTC'},
                    'reminders': {'useDefault': True}
                }
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
                # No free slot, fall back to creating a task
                # (reuse task creation code)
                pass

        # ----- Handle "schedule" with specific time -----
        if task_json.get('start_datetime'):
            calendar_service = build('calendar', 'v3', credentials=creds)
            event = {
                'summary': task_json['content'],
                'description': task_json.get('description', ''),
                'start': {'dateTime': task_json['start_datetime'], 'timeZone': 'UTC'},
                'end': {'dateTime': task_json['start_datetime'], 'timeZone': 'UTC'},
                'reminders': {'useDefault': True}
            }
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

        # ----- Fallback: Google Tasks -----
        tasks_service = build('tasks', 'v1', credentials=creds)
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
            task_body['due'] = task_json['date']
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