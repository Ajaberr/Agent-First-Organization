from ..tools import register_tool, logger
import ast
import hubspot
from hubspot.crm.objects.meetings.models import SimplePublicObjectInputForCreate
from arklex.env.tools.hubspot.utils import HUBSPOT_AUTH_ERROR
from hubspot.crm.objects.meetings import ApiException
import parsedatetime
from datetime import datetime, timedelta
from dateutil import parser
from pprint import pprint
import json
import pytz

description = "Give the customer that the unavailable time of the specific representative and the representative's related meeting link information."

slots = [
    {
        "name": "representative_contact_information",
        "type": "string",
        "description": "The detailed information of the representative (actually a dict) is provided",
        "prompt": "",
        "required": True,
    },
    {
        "name": "time_zone",
        "type": "string",
        "enum": ["America/New_York", "America/Los_Angeles", "Asia/Tokyo", "Europe/London"],
        "description": "The timezone of the user. For example, 'America/New_York'.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True
    },
    {
        "name": "meeting_date",
        "type": "string",
        "description": "The exact date the customer want to take meeting with the representative. e.g. today, Next Monday, May 1st.",
        "prompt": "Could you please give me the date of the meeting?",
        "required": True,
    }
]
outputs = [
    {
        "name": "meeting_link_related_info",
        "type": "string",
        "description": "The unavailable time slots of the representative and the corresponding slug.",
    }
]

MEETING_LINK_UNFOUND_ERROR = "error: the representative does not have a meeting link."

errors = [
    HUBSPOT_AUTH_ERROR,
    MEETING_LINK_UNFOUND_ERROR
]


@register_tool(description, slots, outputs, lambda x: x not in errors)
def check_available(representative_contact_information: str, time_zone: str, meeting_date: str, **kwargs) -> str:
    access_token = kwargs.get('access_token')
    representative_contact_information = ast.literal_eval(representative_contact_information)
    representative_id = representative_contact_information.get('owner_id')
    pprint(f'representative_id: {representative_id}')
    if not access_token:
        return HUBSPOT_AUTH_ERROR
    api_client = hubspot.Client.create(access_token=access_token)
    meeting_link_related_info = {
        'busy_time_slots': [],
        'busy_time_slots_unix': []
    }
    try:
        meeting_link_response = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links",
                "method": "GET",
                "headers": {
                    'Content-Type': 'application/json'
                },
                "qs": {
                    'organizerUserId': representative_id
                }
            }
        )
        meeting_link_response = meeting_link_response.json()
        pprint(f'meeting_link_response: {meeting_link_response}')
        if meeting_link_response.get('total') == 0:
            return MEETING_LINK_UNFOUND_ERROR
        else:
            meeting_links = meeting_link_response['results'][0]
        meeting_slug = meeting_links['slug']
        meeting_link_related_info['slug'] = meeting_slug
        try:
            availability_response = api_client.api_request(
                {
                    "path": "/scheduler/v3/meetings/meeting-links/book/{}".format(meeting_slug),
                    "method": "GET",
                    "headers": {
                        'Content-Type': 'application/json'
                    },
                    "qs": {
                        'timezone': time_zone
                    }
                }
            )
            cal = parsedatetime.Calendar()
            time_struct, _ = cal.parse(meeting_date)
            meeting_date = datetime(*time_struct[:3])
            pprint(f'meeting_date: {meeting_date}')
            availability_response = availability_response.json()
            pprint(f'availability_response: {availability_response}')
            busy_times = availability_response['allUsersBusyTimes'][0]['busyTimes']
            pprint(f'busy_times: {busy_times}')
            for busy_time in busy_times:
                start_time = datetime.fromtimestamp(busy_time["start"] / 1000)
                end_time = datetime.fromtimestamp(busy_time["end"] / 1000)
                pprint(f'busy_time: {busy_time}')
                if start_time.date() == meeting_date.date():
                    meeting_link_related_info['busy_time_slots'].append({
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    })
                    meeting_link_related_info['busy_time_slots_unix'].append({
                        "start": busy_time["start"],
                        "end": busy_time["end"]
                    })
            return str(meeting_link_related_info)
        except ApiException as e:
            pprint(e)
    except ApiException as e:
        print(e)




