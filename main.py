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
model = genai.GenerativeModel("gemini-2.5-flash")  # or gemini-1.5-flash

SCOPES = [
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",  # needed for freebusy query
]
TOKEN_URI = "https://oauth2.googleapis.com/token"


def get_google_credentials():
    creds = Credentials(
        token=None,
        refresh_token=REFRESH_TOKEN,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_uri=TOKEN_URI,
        scopes=SCOPES,
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
        "items": [{"id": "primary"}],
    }
    freebusy = calendar_service.freebusy().query(body=body).execute()
    busy = [
        (datetime.fromisoformat(b["start"]), datetime.fromisoformat(b["end"]))
        for b in freebusy["calendars"]["primary"]["busy"]
    ]
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
            You are a smart assistant that extracts task/event details from user input.
            Today is {current_date} in Sri Lanka (UTC+5:30). The current time is {current_time}.
            The user may speak in Sinhala (සිංහල) or English.

            You must output **only a JSON object** with these fields:
            - "content": (string) the task/event title, keeping original language, removing conversational fluff.
            - "intent": (string) one of:
                * "schedule" – when a specific time is mentioned (e.g., "at 9am").
                * "find_slot" – when the user wants a suitable time (e.g., "find a time", "auto schedule") OR when a date is mentioned without a time (unless the user says "just remind me", "don't schedule", or similar).
                * "task" – for simple tasks with no date/time, or when the user explicitly wants a reminder without scheduling.
            - "date": (string or null) ISO date YYYY-MM-DD if a date is mentioned, else null.
            - "start_datetime": (string or null) ISO datetime in UTC (with Z) if a specific time is given, else null.
            - "duration_minutes": (integer, default 60) estimated duration in minutes, only used when intent is "find_slot".
            - "preferred_time": (string or null) one of "morning", "afternoon", "evening", or null. Used only when intent is "find_slot".
            - "description": (string or null) any extra notes, else null.
            - "labels": (list of strings or null) any labels/projects mentioned, else null.

            Interpretation rules:
            - Use today's date ({current_date}) as the reference for relative words like "today", "tomorrow", "next Friday".
            - All times are given in Sri Lanka time (UTC+5:30). Convert them to UTC for start_datetime (e.g., 9am Sri Lanka = 03:30Z).
            - For "find_slot", we will later query your calendar and pick a free time on the specified date (or today if none). If no date is provided, default to tomorrow.
            - If a date is mentioned but no time, set intent to "find_slot" **unless** the user indicates they only want a reminder (e.g., "remind me", "just remind", "don't schedule").
            - If both start_datetime and date are present, use start_datetime and set date to null.
            - Duration and preferred_time are optional hints for "find_slot". Use reasonable defaults: duration 60 minutes if not specified; preferred_time from words like "morning", "afternoon", "evening".
            - If the user asks for a specific time but also says "find a good time", prefer "schedule" (they gave the time).

            Examples:

            1. "I have a meeting tomorrow at 9.00 am"
            → {{"content": "Meeting", "intent": "schedule", "date": null, "start_datetime": "2026-03-24T03:30:00Z", "duration_minutes": null, "preferred_time": null, "description": "I have a meeting", "labels": null}}

            2. "හෙට සුදුසු වෙලාවක් බලලා මට ගෙදර වැඩ කරන්න දාන්න"
            → {{"content": "ගෙදර වැඩ", "intent": "find_slot", "date": "2026-03-24", "start_datetime": null, "duration_minutes": 60, "preferred_time": null, "description": null, "labels": null}}

            3. "හෙට මට පාඩම් කරන්න ඕනි."
            → {{"content": "පාඩම් කරන්න", "intent": "find_slot", "date": "2026-03-24", "start_datetime": null, "duration_minutes": 60, "preferred_time": null, "description": null, "labels": null}}

            4. "Buy milk tomorrow"
            → {{"content": "Buy milk", "intent": "find_slot", "date": "2026-03-24", "start_datetime": null, "duration_minutes": 60, "preferred_time": null, "description": null, "labels": null}}

            5. "Remind me to call John tomorrow" (explicit reminder)
            → {{"content": "Call John", "intent": "task", "date": "2026-03-24", "start_datetime": null, "duration_minutes": null, "preferred_time": null, "description": "Remind me to call John", "labels": null}}

            6. "Call John"
            → {{"content": "Call John", "intent": "task", "date": null, "start_datetime": null, "duration_minutes": null, "preferred_time": null, "description": null, "labels": null}}

            7. "Schedule a 30-minute reading session tomorrow afternoon"
            → {{"content": "Reading session", "intent": "find_slot", "date": "2026-03-24", "start_datetime": null, "duration_minutes": 30, "preferred_time": "afternoon", "description": null, "labels": null}}

            8. "I have a dentist appointment on Friday at 2pm"
            → {{"content": "Dentist appointment", "intent": "schedule", "date": null, "start_datetime": "2026-03-27T08:30:00Z", "duration_minutes": null, "preferred_time": null, "description": null, "labels": null}}

            Now, process the user's text and output only the JSON object (no other text).

            Text: {item.text}
        """

        response = model.generate_content(prompt)
        task_data_str = response.text.strip()
        if task_data_str.startswith("```json"):
            task_data_str = task_data_str[7:-3].strip()
        task_json = json.loads(task_data_str)
        print("Gemini output:", json.dumps(task_json, indent=2))

        creds = get_google_credentials()
        intent = task_json.get("intent", "task")

        # ----- Handle "find_slot" -----
        if intent == "find_slot":
            date_to_check = task_json.get("date")
            if not date_to_check:
                tomorrow = datetime.now(sri_lanka_tz) + timedelta(days=1)
                date_to_check = tomorrow.strftime("%Y-%m-%d")
            duration = task_json.get("duration_minutes", 60)
            calendar_service = build("calendar", "v3", credentials=creds)
            free_slots = get_free_slots(calendar_service, date_to_check, duration)
            print("Free slots:", free_slots)
            if free_slots:
                chosen_start, chosen_end = free_slots[0]
                # Format without microseconds, with 'Z'
                start_iso = chosen_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_iso = chosen_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                event = {
                    "summary": task_json["content"],
                    "description": task_json.get("description", ""),
                    "start": {"dateTime": start_iso, "timeZone": "UTC"},
                    "end": {"dateTime": end_iso, "timeZone": "UTC"},
                    "reminders": {"useDefault": True},
                }
                if task_json.get("labels"):
                    event[
                        "description"
                    ] += f"\n\nLabels: {', '.join(task_json['labels'])}"
                print("Event body:", json.dumps(event, indent=2))
                created_event = (
                    calendar_service.events()
                    .insert(calendarId="primary", body=event)
                    .execute()
                )
                print("Calendar event created:", json.dumps(created_event, indent=2))
                return {
                    "status": "success",
                    "event": {
                        "id": created_event["id"],
                        "title": created_event["summary"],
                        "link": created_event.get("htmlLink"),
                    },
                }
            else:
                # No free slot found – fall back to creating a task
                # (You can reuse the tasks code here)
                pass
       
        # ----- Handle "schedule" with specific time -----
        if task_json.get("start_datetime"):
            calendar_service = build("calendar", "v3", credentials=creds)
            event = {
                "summary": task_json["content"],
                "description": task_json.get("description", ""),
                "start": {"dateTime": task_json["start_datetime"], "timeZone": "UTC"},
                "end": {"dateTime": task_json["start_datetime"], "timeZone": "UTC"},
                "reminders": {"useDefault": True},
            }
            if task_json.get("labels"):
                event["description"] += f"\n\nLabels: {', '.join(task_json['labels'])}"
            created_event = (
                calendar_service.events()
                .insert(calendarId="primary", body=event)
                .execute()
            )
            print("Calendar event created:", json.dumps(created_event, indent=2))
            return {
                "status": "success",
                "event": {
                    "id": created_event["id"],
                    "title": created_event["summary"],
                    "link": created_event.get("htmlLink"),
                },
            }

        # ----- Fallback: Google Tasks -----
        tasks_service = build("tasks", "v1", credentials=creds)
        task_lists = tasks_service.tasklists().list().execute()
        if not task_lists.get("items"):
            default_list = (
                tasks_service.tasklists().insert(body={"title": "My Tasks"}).execute()
            )
            task_list_id = default_list["id"]
        else:
            task_list_id = task_lists["items"][0]["id"]
        task_body = {
            "title": task_json["content"],
            "notes": task_json.get("description", ""),
        }
        if task_json.get("date"):
            task_body["due"] = task_json["date"]
        if task_json.get("labels"):
            labels_text = f"\n\nLabels: {', '.join(task_json['labels'])}"
            task_body["notes"] = (task_body.get("notes", "") + labels_text).strip()
        created_task = (
            tasks_service.tasks()
            .insert(tasklist=task_list_id, body=task_body)
            .execute()
        )
        return {
            "status": "success",
            "task": {"id": created_task["id"], "title": created_task["title"]},
        }

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"status": "ok"}
