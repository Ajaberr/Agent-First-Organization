from ..tools import register_tool, logger
import ast
import hubspot
from arklex.env.tools.hubspot.utils import HUBSPOT_AUTH_ERROR
from hubspot.crm.owners import ApiException
from datetime import datetime
from pprint import pprint


description = "Find the detailed information about the owner."


slots = [
    {
        "name": "full_name",
        "type": "string",
        "description": "The full name of the owner",
        "prompt": "Can you tell me who you want to contact (full name)? e.g. Veronica Chen. (not first name like Veronica)",
        "required": True,
    }
]
outputs = [
    {
        "name": "owner_information",
        "type": "string",
        "description": "The detailed information about the owner (actually is a dict)",
    }
]

OWNER_NOT_FOUND_ERROR = "error: the owner is not found"
errors = [
    HUBSPOT_AUTH_ERROR,
    OWNER_NOT_FOUND_ERROR
]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def find_owner_by_full_name(full_name: str, **kwargs) -> str:
    access_token = kwargs.get('access_token')

    if not access_token:
        return HUBSPOT_AUTH_ERROR

    api_client = hubspot.Client.create(access_token=access_token)

    '''
    Get the entire list of the owners
    '''
    try:
        owners_response = api_client.crm.owners.owners_api.get_page()
        pprint(owners_response)
        owners_list = owners_response._results
        index = next((i for i, owner in enumerate(owners_list) if f"{owner.first_name} {owner.last_name}" == full_name), -1)

        if index != -1:
            owner_information = {
                "owner_id": owners_list[index].id,
                "owner_first_name": owners_list[index].first_name,
                "owner_last_name": owners_list[index].last_name,
                "owner_email": owners_list[index].email
            }

            pprint(owners_response)
            return str(owner_information)
        else:
            return OWNER_NOT_FOUND_ERROR
    except ApiException as e:
        logger.info("Exception when calling owner_api: %s\n" % e)
