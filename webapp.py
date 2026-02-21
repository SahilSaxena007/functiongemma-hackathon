import atexit
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, time as time_obj
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, render_template, request

from main import generate_hybrid
from cactus import cactus_destroy, cactus_init, cactus_transcribe

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "America/Los_Angeles")
CALENDAR_ID = os.environ.get("GOOGLE_CALENDAR_ID", "primary")
GOOGLE_CREDENTIALS_PATH = os.environ.get("GOOGLE_OAUTH_CREDENTIALS", "credentials.json")
GOOGLE_TOKEN_PATH = os.environ.get("GOOGLE_OAUTH_TOKEN", ".secrets/google_token.json")

WHISPER_MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", "cactus/weights/whisper-small")
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


AGENT_TOOLS = [
    {
        "name": "create_calendar_event",
        "description": "Create a Google Calendar event from date/time details.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date": {"type": "string", "description": "Date such as today, tomorrow, Monday, YYYY-MM-DD"},
                "time": {"type": "string", "description": "Time such as 2:30 PM or 14:30"},
                "duration_minutes": {"type": "integer", "description": "Event length in minutes"},
                "description": {"type": "string", "description": "Optional details"},
            },
            "required": ["title", "date", "time"],
        },
    },
    {
        "name": "reschedule_calendar_event",
        "description": "Reschedule an existing calendar event found by a query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to find the event, like dentist or team sync"},
                "new_date": {"type": "string", "description": "New date"},
                "new_time": {"type": "string", "description": "New time"},
                "duration_minutes": {"type": "integer", "description": "Optional new duration in minutes"},
            },
            "required": ["query", "new_date", "new_time"],
        },
    },
    {
        "name": "delete_calendar_event",
        "description": "Delete a calendar event found by a query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to find the event"},
                "date": {"type": "string", "description": "Optional date constraint"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_calendar_events",
        "description": "List events on a given date.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date like today, tomorrow, Monday, YYYY-MM-DD"},
            },
            "required": [],
        },
    },
]

_WHISPER = None
_GOOGLE_SERVICE = None


def _ensure_google_service():
    global _GOOGLE_SERVICE
    if _GOOGLE_SERVICE is not None:
        return _GOOGLE_SERVICE

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except Exception as exc:
        raise RuntimeError(
            "Google Calendar dependencies missing. Install: "
            "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    scopes = ["https://www.googleapis.com/auth/calendar"]
    creds = None
    if os.path.exists(GOOGLE_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_PATH, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
                raise RuntimeError(
                    f"Missing OAuth credentials file at '{GOOGLE_CREDENTIALS_PATH}'. "
                    "Set GOOGLE_OAUTH_CREDENTIALS or place credentials.json in repo root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_PATH, scopes)
            creds = flow.run_local_server(port=0)

        token_dir = os.path.dirname(GOOGLE_TOKEN_PATH)
        if token_dir:
            os.makedirs(token_dir, exist_ok=True)
        with open(GOOGLE_TOKEN_PATH, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    _GOOGLE_SERVICE = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _GOOGLE_SERVICE


def _get_whisper():
    global _WHISPER
    if _WHISPER is None:
        _WHISPER = cactus_init(WHISPER_MODEL_PATH)
    return _WHISPER


def _destroy_whisper():
    global _WHISPER
    if _WHISPER is not None:
        try:
            cactus_destroy(_WHISPER)
        finally:
            _WHISPER = None


atexit.register(_destroy_whisper)


def _resolve_date(date_text, timezone_name):
    now = datetime.now(ZoneInfo(timezone_name))
    if not date_text:
        return now.date()

    text = date_text.strip().lower()
    if text in {"today", "tonight"}:
        return now.date()
    if text == "tomorrow":
        return (now + timedelta(days=1)).date()
    if text == "day after tomorrow":
        return (now + timedelta(days=2)).date()

    iso_match = re.match(r"^\d{4}-\d{2}-\d{2}$", text)
    if iso_match:
        return datetime.strptime(text, "%Y-%m-%d").date()

    slash_match = re.match(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$", text)
    if slash_match:
        month = int(slash_match.group(1))
        day = int(slash_match.group(2))
        year_group = slash_match.group(3)
        year = int(year_group) if year_group else now.year
        if year < 100:
            year += 2000
        return datetime(year, month, day).date()

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    next_prefix = text.startswith("next ")
    weekday_name = text.replace("next ", "", 1).strip()
    if weekday_name in weekdays:
        target = weekdays[weekday_name]
        delta = (target - now.weekday()) % 7
        if next_prefix:
            delta = delta + 7 if delta != 0 else 7
        return (now + timedelta(days=delta)).date()

    return now.date()


def _resolve_time(time_text):
    if not time_text:
        return time_obj(9, 0)

    text = time_text.strip().lower()
    ampm_match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", text)
    if ampm_match:
        hour = int(ampm_match.group(1))
        minute = int(ampm_match.group(2) or 0)
        meridiem = ampm_match.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return time_obj(hour, minute)

    hm_match = re.match(r"^(\d{1,2})(?::(\d{2}))?$", text)
    if hm_match:
        hour = int(hm_match.group(1))
        minute = int(hm_match.group(2) or 0)
        hour = max(0, min(hour, 23))
        minute = max(0, min(minute, 59))
        return time_obj(hour, minute)

    return time_obj(9, 0)


def _build_dt(date_text, time_text, timezone_name):
    d = _resolve_date(date_text, timezone_name)
    t = _resolve_time(time_text)
    return datetime.combine(d, t, tzinfo=ZoneInfo(timezone_name))


def _event_dt(event, field):
    block = event.get(field, {})
    if "dateTime" in block:
        return datetime.fromisoformat(block["dateTime"].replace("Z", "+00:00"))
    if "date" in block:
        return datetime.fromisoformat(f"{block['date']}T00:00:00")
    return None


def _event_duration_minutes(event):
    start = _event_dt(event, "start")
    end = _event_dt(event, "end")
    if start and end:
        delta = end - start
        return max(1, int(delta.total_seconds() // 60))
    return 60


def _list_events(service, time_min, time_max, query=None, max_results=20):
    return service.events().list(
        calendarId=CALENDAR_ID,
        timeMin=time_min.isoformat(),
        timeMax=time_max.isoformat(),
        singleEvents=True,
        orderBy="startTime",
        q=query,
        maxResults=max_results,
    ).execute().get("items", [])


def _find_event(service, query, date_text=None):
    tz = ZoneInfo(APP_TIMEZONE)
    if date_text:
        day = _resolve_date(date_text, APP_TIMEZONE)
        start = datetime.combine(day, time_obj.min, tzinfo=tz)
        end = start + timedelta(days=1)
    else:
        now = datetime.now(tz)
        start = now - timedelta(days=1)
        end = now + timedelta(days=120)

    candidates = _list_events(service, start, end, query=query, max_results=20)
    if not candidates:
        return None

    if query:
        query_lower = query.lower()
        for event in candidates:
            summary = (event.get("summary") or "").lower()
            if query_lower in summary:
                return event
    return candidates[0]


def _normalize_event(event):
    start_block = event.get("start", {})
    end_block = event.get("end", {})
    return {
        "id": event.get("id"),
        "summary": event.get("summary"),
        "start": start_block.get("dateTime") or start_block.get("date"),
        "end": end_block.get("dateTime") or end_block.get("date"),
        "html_link": event.get("htmlLink"),
    }


def _execute_tool_call(call):
    name = call.get("name")
    args = call.get("arguments", {}) or {}
    service = _ensure_google_service()

    if name == "create_calendar_event":
        title = str(args.get("title", "New event")).strip() or "New event"
        date_text = str(args.get("date", "today"))
        time_text = str(args.get("time", "9:00 AM"))
        duration = int(args.get("duration_minutes", 60))
        description = str(args.get("description", "")).strip()

        start_dt = _build_dt(date_text, time_text, APP_TIMEZONE)
        end_dt = start_dt + timedelta(minutes=max(1, duration))
        payload = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": APP_TIMEZONE},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": APP_TIMEZONE},
        }
        created = service.events().insert(calendarId=CALENDAR_ID, body=payload).execute()
        return {"ok": True, "tool": name, "result": _normalize_event(created)}

    if name == "reschedule_calendar_event":
        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "tool": name, "error": "Missing query."}

        event = _find_event(service, query=query, date_text=args.get("date"))
        if not event:
            return {"ok": False, "tool": name, "error": f"No event found for '{query}'."}

        new_date = str(args.get("new_date", "today"))
        new_time = str(args.get("new_time", "9:00 AM"))
        duration = args.get("duration_minutes")
        duration_minutes = int(duration) if duration is not None else _event_duration_minutes(event)

        start_dt = _build_dt(new_date, new_time, APP_TIMEZONE)
        end_dt = start_dt + timedelta(minutes=max(1, duration_minutes))
        event["start"] = {"dateTime": start_dt.isoformat(), "timeZone": APP_TIMEZONE}
        event["end"] = {"dateTime": end_dt.isoformat(), "timeZone": APP_TIMEZONE}
        updated = service.events().update(calendarId=CALENDAR_ID, eventId=event["id"], body=event).execute()
        return {"ok": True, "tool": name, "result": _normalize_event(updated)}

    if name == "delete_calendar_event":
        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "tool": name, "error": "Missing query."}
        event = _find_event(service, query=query, date_text=args.get("date"))
        if not event:
            return {"ok": False, "tool": name, "error": f"No event found for '{query}'."}

        service.events().delete(calendarId=CALENDAR_ID, eventId=event["id"]).execute()
        return {"ok": True, "tool": name, "result": {"deleted": _normalize_event(event)}}

    if name == "list_calendar_events":
        date_text = str(args.get("date", "today"))
        day = _resolve_date(date_text, APP_TIMEZONE)
        tz = ZoneInfo(APP_TIMEZONE)
        start = datetime.combine(day, time_obj.min, tzinfo=tz)
        end = start + timedelta(days=1)
        events = _list_events(service, start, end, query=None, max_results=25)
        return {"ok": True, "tool": name, "result": {"date": str(day), "events": [_normalize_event(e) for e in events]}}

    return {"ok": False, "tool": name or "unknown", "error": "Unsupported tool call."}


def _run_agent(text):
    model_messages = [{"role": "user", "content": text}]
    result = generate_hybrid(model_messages, AGENT_TOOLS)
    calls = result.get("function_calls", []) or []
    executions = []
    for call in calls:
        try:
            executions.append(_execute_tool_call(call))
        except Exception as exc:
            executions.append({"ok": False, "tool": call.get("name"), "error": str(exc)})

    return {
        "source": result.get("source", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "total_time_ms": result.get("total_time_ms", 0.0),
        "function_calls": calls,
        "executions": executions,
    }


def _transcribe_audio_wav(path):
    model = _get_whisper()
    raw = cactus_transcribe(model, path, prompt=WHISPER_PROMPT)
    data = json.loads(raw)
    transcript = (data.get("response") or "").strip()
    if not transcript:
        raise RuntimeError("Empty transcript from cactus_transcribe.")
    return transcript


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/text-act")
def api_text_act():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "error": "Message is required."}), 400

    try:
        result = _run_agent(message)
        return jsonify({"ok": True, "input": message, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/voice-act")
def api_voice_act():
    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "audio file is required (WAV)."}), 400

    audio = request.files["audio"]
    if not audio.filename:
        return jsonify({"ok": False, "error": "Empty filename."}), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name
            audio.save(temp_path)

        transcript = _transcribe_audio_wav(temp_path)
        result = _run_agent(transcript)
        return jsonify({"ok": True, "transcript": transcript, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
