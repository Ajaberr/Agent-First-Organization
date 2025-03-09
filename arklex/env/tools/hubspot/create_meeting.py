from ..tools import register_tool, logger
import ast
import hubspot
from hubspot.crm.objects.meetings.models import SimplePublicObjectInputForCreate
from arklex.env.tools.hubspot.utils import HUBSPOT_AUTH_ERROR
from hubspot.crm.objects.meetings import ApiException
import parsedatetime
from datetime import datetime
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
        "name": "representative_contact_information",
        "type": "string",
        "description": "The detailed information of the representative (actually a dict) is provided",
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
        "prompt": "Could you please give me the start time of the meeting?",
        "required": True,
    },
    {
        "name": "meeting_end_time",
        "type": "string",
        "description": "The exact end time the customer want to take meeting with the representative",
        "prompt": "Could you please give me the end time of the meeting?",
        "required": True,
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
def create_meeting(customer_contact_information: str, representative_contact_information: str, meeting_date: str, meeting_start_time: str, meeting_end_time: str, **kwargs) -> str:
    access_token = kwargs.get('access_token')
    customer_contact_information = ast.literal_eval(customer_contact_information)
    customer_contact_id = customer_contact_information.get('id')

    meeting_date = parse_natural_date(meeting_date, timezone="America/New_York")

    meeting_start_time = parse_natural_date(meeting_start_time, meeting_date, timezone="America/New_York")

    meeting_end_time = parse_natural_date(meeting_end_time, meeting_start_time, timezone="America/New_York")

    meeting_start_time = meeting_start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    meeting_end_time = meeting_end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    pprint(f"meeting_start_time: {meeting_start_time}")
    pprint(f"meeting_end_time: {meeting_end_time}")

    # customer_name = customer_contact_information.get('firstname')

    representative_contact_information = ast.literal_eval(representative_contact_information)
    representative_id = representative_contact_information.get('owner_id')
    # representative_name = representative_contact_information.get('firstname')


    if not access_token:
        return HUBSPOT_AUTH_ERROR

    api_client = hubspot.Client.create(access_token=access_token)
    meeting_properties = {
        "hs_meeting_title": f"Meeting between {representative_id} and {customer_contact_id} at {meeting_start_time}",
        "hubspot_owner_id": representative_id,
        "hs_timestamp": meeting_start_time,
        "hs_meeting_start_time": meeting_start_time,
        "hs_meeting_end_time": meeting_end_time,
        "hs_meeting_outcome": "SCHEDULED",
        "hs_meeting_location": "Remote"
    }
    associaion_info = [
        {
            "to": {
                "id": customer_contact_id
            },
            "types": [
                {
                    "associationCategory": "HUBSPOT_DEFINED",
                    "associationTypeId": 200
                }
            ]
        }
    ]
    meeting_information = SimplePublicObjectInputForCreate(
        properties=meeting_properties, associations=associaion_info
    )
    try:
        meeting_creation_response = api_client.crm.objects.meetings.basic_api.create(
            simple_public_object_input_for_create=meeting_information)
        pprint(meeting_creation_response)
        meeting_creation_response = meeting_creation_response.to_dict()
        return meeting_creation_response
    except ApiException as e:
        logger.info("Exception when calling Crm.tickets.create: %s\n" % e)


def parse_natural_date(date_str, base_date=None, timezone=None):
    cal = parsedatetime.Calendar()
    time_struct, _ = cal.parse(date_str, base_date)
    parsed_dt = datetime(*time_struct[:6])

    if base_date and (parsed_dt.date() != base_date.date()):
        parsed_dt = datetime.combine(base_date.date(), parsed_dt.time())

    # Handle time zone if provided
    if timezone:
        local_timezone = pytz.timezone(timezone)
        parsed_dt = local_timezone.localize(parsed_dt)  # Localize to the specified timezone
        parsed_dt = parsed_dt.astimezone(pytz.utc)  # Convert to UTC

    return parsed_dt


