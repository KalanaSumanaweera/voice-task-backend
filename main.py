import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

load_dotenv()

# ---------- Groq configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # or mixtral-8x7b-32768, etc.
client = Groq(api_key=GROQ_API_KEY)

# ---------- Google OAuth ----------
REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

SCOPES = [
    'https://www.googleapis.com/auth/tasks',
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/calendar.readonly'
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

        # Compute tomorrow's date for use in examples
        tomorrow = (datetime.fromisoformat(current_date) + timedelta(days=1)).strftime("%Y-%m-%d")

        # ---------- DYNAMIC PROMPT with computed dates ----------
        system_message = f"""
You are a smart assistant that extracts task/event details from user input.
Today is {current_date} in Sri Lanka (UTC+5:30). The current time is {current_time}.
The user may speak in Sinhala (සිංහල) or English.

You must output **only a JSON object** with these fields:
- "content": (string) the task/event title, keeping original language, removing conversational fluff.
- "intent": (string) one of:
    * "schedule" – when a specific time is mentioned (e.g., "at 9am").
    * "find_slot" – when the user wants a suitable time (e.g., "find a time", "auto schedule") OR when a date is mentioned without a time (unless the user says "just remind me", "don't schedule", or similar).
    * "task" – for simple tasks with no date/time, or when the user explicitly wants a reminder without scheduling.
    * "query" – when the user asks about existing plans (e.g., "what are my plans today", "show me tasks for tomorrow").
- "date": (string or null) ISO date YYYY-MM-DD if a date is mentioned, else null.
- "start_datetime": (string or null) ISO datetime in UTC (with Z) if a specific time is given, else null.
- "duration_minutes": (integer, default 60) estimated duration in minutes, only used when intent is "find_slot".
- "preferred_time": (string or null) one of "morning", "afternoon", "evening", or null.
- "description": (string or null) any extra notes, else null.
- "labels": (list of strings or null) any labels/projects mentioned, else null.
- "language": (string) the language of the user's input, either "si" (Sinhala) or "en" (English).

Date/Time Calculation:
- Use today's date {current_date} as the base.
- For "tomorrow", add 1 day to {current_date} → {tomorrow}.
- For "next Friday", compute the next Friday after {current_date}.
- For any relative term, calculate the exact date.
- All times are in Sri Lanka time (UTC+5:30). Convert to UTC for start_datetime (e.g., 9am Sri Lanka = 03:30Z).
- If only a date is mentioned, set "date" to that date, and leave "start_datetime" null.
- If a specific time is mentioned (e.g., "at 9am"), set "start_datetime" to the full datetime in UTC.

Examples (using today's date {current_date}):

1. "I have a meeting tomorrow at 9.00 am"
   → Output: {{"content": "Meeting", "intent": "schedule", "date": null, "start_datetime": "{tomorrow}T03:30:00Z", "duration_minutes": null, "preferred_time": null, "description": "I have a meeting", "labels": null, "language": "en"}}

2. "හෙට සුදුසු වෙලාවක් බලලා මට ගෙදර වැඩ කරන්න දාන්න"
   → Output: {{"content": "ගෙදර වැඩ", "intent": "find_slot", "date": "{tomorrow}", "start_datetime": null, "duration_minutes": 60, "preferred_time": null, "description": null, "labels": null, "language": "si"}}

3. "Buy milk tomorrow"
   → Output: {{"content": "Buy milk", "intent": "find_slot", "date": "{tomorrow}", "start_datetime": null, "duration_minutes": 60, "preferred_time": null, "description": null, "labels": null, "language": "en"}}

4. "What are my plans today?"
   → Output: {{"content": null, "intent": "query", "date": "{current_date}", "start_datetime": null, "duration_minutes": null, "preferred_time": null, "description": null, "labels": null, "language": "en"}}

5. "අද මගේ සැලසුම් මොනවාද?"
   → Output: {{"content": null, "intent": "query", "date": "{current_date}", "start_datetime": null, "duration_minutes": null, "preferred_time": null, "description": null, "labels": null, "language": "si"}}

Now, process the user's text and output only the JSON object (no other text). Ensure dates are correctly computed based on today's date {current_date}.
"""

        user_message = item.text

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        task_data_str = response.choices[0].message.content.strip()
        # Remove markdown code fences if present
        if task_data_str.startswith("```json"):
            task_data_str = task_data_str[7:-3].strip()
        task_json = json.loads(task_data_str)
        print("Groq output:", json.dumps(task_json, indent=2))

        creds = get_google_credentials()
        intent = task_json.get('intent', 'task')

        # ---------- QUERY INTENT ----------
        if intent == 'query':
            target_date = task_json.get('date')
            if not target_date:
                target_date = datetime.now(sri_lanka_tz).strftime("%Y-%m-%d")
            language = task_json.get('language', 'en')

            calendar_service = build('calendar', 'v3', credentials=creds)
            start_of_day = datetime.fromisoformat(f"{target_date}T00:00:00+00:00")
            end_of_day = datetime.fromisoformat(f"{target_date}T23:59:59+00:00")
            events_result = calendar_service.events().list(
                calendarId='primary',
                timeMin=start_of_day.isoformat(),
                timeMax=end_of_day.isoformat(),
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])

            tasks_service = build('tasks', 'v1', credentials=creds)
            task_lists = tasks_service.tasklists().list().execute()
            filtered_tasks = []
            if task_lists.get('items'):
                default_list_id = task_lists['items'][0]['id']
                tasks_result = tasks_service.tasks().list(tasklist=default_list_id, showHidden=False, maxResults=100).execute()
                tasks = tasks_result.get('items', [])
                filtered_tasks = [t for t in tasks if t.get('due') and t['due'][:10] == target_date]

            events_summary = []
            for e in events:
                start = e['start'].get('dateTime', e['start'].get('date'))
                events_summary.append({"title": e['summary'], "start": start})
            tasks_summary = [{"title": t['title']} for t in filtered_tasks]

            summary_prompt = f"""
The user asked in language code {language} about their plans for {target_date}.
Events on that date: {json.dumps(events_summary, ensure_ascii=False)}
Tasks due on that date: {json.dumps(tasks_summary, ensure_ascii=False)}

Please produce a friendly spoken summary in the same language (language code {language}). If no events or tasks, say "You have no plans" in that language.
Return only the summary text, no extra formatting.
"""
            summary_response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes plans."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            summary_text = summary_response.choices[0].message.content.strip()
            print("Summary:", summary_text)

            return {
                "status": "success",
                "query_result": summary_text,
                "language": language
            }

        # ---------- FIND_SLOT ----------
        if intent == 'find_slot':
            date_to_check = task_json.get('date')
            if not date_to_check:
                tomorrow = datetime.now(sri_lanka_tz) + timedelta(days=1)
                date_to_check = tomorrow.strftime("%Y-%m-%d")
            duration = task_json.get('duration_minutes', 60)
            calendar_service = build('calendar', 'v3', credentials=creds)
            free_slots = get_free_slots(calendar_service, date_to_check, duration)
            if free_slots:
                chosen_start, chosen_end = free_slots[0]
                start_iso = chosen_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_iso = chosen_end.strftime("%Y-%m-%dT%H:%M:%SZ")
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
                # Fall through to task creation
                pass

        # ---------- SCHEDULE (exact time) ----------
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

        # ---------- FALLBACK: Google Tasks ----------
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
