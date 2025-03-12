import json

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
import pytz

description = "Schedule a meeting for the existing customer with the specific representative."


slots = [
    {
        "name": "customer_contact_information",
        "type": "string",
        "description": "After finding the exiting customer, the detailed information of the customer (actually a dict) is provided",
        "prompt": "",
        "required": True,
    },
    {
        "name": "meeting_date",
        "type": "string",
        "description": "The exact date the customer want to take meeting with the representative. e.g. today, Next Monday, May 1st.",
        "prompt": "Could you please give me the date of the meeting?",
        "required": True,
    },
    {
        "name": "meeting_start_time",
        "type": "string",
        "description": "The exact start time the customer want to take meeting with the representative. e.g. 1pm",
        "prompt": "Could you please give me the start time of the meeting? Typically, the representative will hold the meeting from 9:00 am to 4:45 pm.",
        "required": True,
    },
    {
        "name": "duration",
        "type": "integer",
        "enum": [15, 30, 60],
        "description": "The exact duration of the meeting. Please ask the user to input. DO NOT AUTOMATICALLY GIVE THE SLOT ANY VALUE.",
        "prompt": "Could you please give me the duration of the meeting (e.g. 15, 30, 60 mins)?",
        "required": True,
    },
    {
        "name": "meeting_link_related_info",
        "type": "string",
        "description": "The unavailable time slots of the representative and the corresponding slug. This is actually a dict.",
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
    }
]
outputs = [
    {
        "name": "meeting_confirmation_info",
        "type": "string",
        "description": "The detailed information about the meeting to let the customer confirm",
    }
]

UNAVAILABLE_ERROR = "error: the representative is not available during the required period."

errors = [
    HUBSPOT_AUTH_ERROR,
    UNAVAILABLE_ERROR
]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def create_meeting(customer_contact_information: str,meeting_date: str,
                   meeting_start_time: str, duration: int,
                   meeting_link_related_info: str, time_zone: str, **kwargs) -> str:
    access_token = kwargs.get('access_token')
    if not access_token:
        return HUBSPOT_AUTH_ERROR

    customer_contact_information = ast.literal_eval(customer_contact_information)
    customer_id = customer_contact_information.get('contact_id')
    customer_first_name = customer_contact_information.get('contact_first_name')
    customer_last_name = customer_contact_information.get('contact_last_name')
    customer_email = customer_contact_information.get('contact_email')


    meeting_date = parse_natural_date(meeting_date, timezone=time_zone, date_input=True)
    pprint('meeting_date: {}'.format(meeting_date))
    meeting_start_time = parse_natural_date(meeting_start_time, meeting_date, timezone=time_zone)
    pprint('meeting_start_time: {}'.format(meeting_start_time))
    meeting_start_time = int(meeting_start_time.timestamp() * 1000)
    pprint('meeting_start_time: {}'.format(meeting_start_time))


    duration = int(duration)
    duration = int(timedelta(minutes=duration).total_seconds() * 1000)

    meeting_link_related_info = ast.literal_eval(meeting_link_related_info)
    meeting_slug = meeting_link_related_info.get('slug')
    unavailable_time_slots = meeting_link_related_info.get('busy_time_slots_unix')

    meeting_end_time = meeting_start_time + duration
    for time_slot in unavailable_time_slots:
        if meeting_start_time >= time_slot['start'] and meeting_start_time < time_slot['end']:
            return UNAVAILABLE_ERROR
        elif meeting_end_time >= time_slot['start'] and meeting_end_time <= time_slot['end']:
            return UNAVAILABLE_ERROR

    api_client = hubspot.Client.create(access_token=access_token)

    try:
        create_meeting_response = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links/book",
                "method": "POST",
                "body": {
                    "slug": meeting_slug,
                    "duration": duration,
                    "startTime": meeting_start_time,
                    "email": customer_email,
                    "firstName": customer_first_name,
                    "lastName": customer_last_name,
                    "timezone": time_zone,
                    "locale": "en-us",
                },
                "qs": {
                    'timezone': time_zone
                }
            }

        )
        create_meeting_response = create_meeting_response.json()
        pprint(create_meeting_response)
        return json.dumps(create_meeting_response)
    except ApiException as e:
        print(e)
    # meeting_properties = {
    #     "hs_meeting_title": f"Meeting between {representative_id} and {customer_contact_id} at {meeting_start_time}",
    #     "hubspot_owner_id": representative_id,
    #     "hs_timestamp": meeting_start_time,
    #     "hs_meeting_start_time": meeting_start_time,
    #     "hs_meeting_end_time": meeting_end_time,
    #     "hs_meeting_outcome": "SCHEDULED",
    #     "hs_meeting_location": "Remote"
    # }
    # associaion_info = [
    #     {
    #         "to": {
    #             "id": customer_contact_id
    #         },
    #         "types": [
    #             {
    #                 "associationCategory": "HUBSPOT_DEFINED",
    #                 "associationTypeId": 200
    #             }
    #         ]
    #     }
    # ]
    # meeting_information = SimplePublicObjectInputForCreate(
    #     properties=meeting_properties, associations=associaion_info
    # )
    # try:
    #     meeting_creation_response = api_client.crm.objects.meetings.basic_api.create(
    #         simple_public_object_input_for_create=meeting_information)
    #     pprint(meeting_creation_response)
    #     meeting_creation_response = meeting_creation_response.to_dict()
    #     return meeting_creation_response
    # except ApiException as e:
    #     logger.info("Exception when calling Crm.tickets.create: %s\n" % e)


def parse_natural_date(date_str, base_date=None, timezone=None, date_input=False):
    cal = parsedatetime.Calendar()
    time_struct, _ = cal.parse(date_str, base_date)
    if date_input:
        parsed_dt = datetime(*time_struct[:3])
    else:
        parsed_dt = datetime(*time_struct[:6])

    pprint('parsed_dt: {}'.format(parsed_dt))
    if base_date and (parsed_dt.date() != base_date.date()):
        parsed_dt = datetime.combine(base_date.date(), parsed_dt.time())
    pprint('first if: parsed_dt: {}'.format(parsed_dt))
    # Handle time zone if provided
    if timezone:
        local_timezone = pytz.timezone(timezone)
        parsed_dt = local_timezone.localize(parsed_dt)  # Localize to the specified timezone
        parsed_dt = parsed_dt.astimezone(pytz.utc)  # Convert to UTC
    pprint('second if: parsed_dt: {}'.format(parsed_dt))
    return parsed_dt


