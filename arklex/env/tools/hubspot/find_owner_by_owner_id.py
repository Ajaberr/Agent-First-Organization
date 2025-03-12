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
        "name": "owner_id",
        "type": "string",
        "description": "The owner id of the owner, extracting from the tool \'find_owner_id_by_contact_id\'(It is actually a dict)",
        "prompt": "",
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
def find_owner_by_owner_id(owner_id: str, **kwargs) -> str:
    access_token = kwargs.get('access_token')

    if not access_token:
        return HUBSPOT_AUTH_ERROR
    api_client = hubspot.Client.create(access_token=access_token)

    '''
    Get the entire list of the owners
    '''
    try:
        get_owner_response = api_client.crm.owners.owners_api.get_by_id(owner_id=owner_id)
        get_owner_response = get_owner_response.to_dict()
        owner_information = {
            "owner_id": get_owner_response['id'],
            "owner_first_name": get_owner_response['first_name'],
            "owner_last_name": get_owner_response['last_name'],
            "owner_email": get_owner_response['email']
        }
        pprint(get_owner_response)
        return str(owner_information)
        # owners_list = get_owner_response._results
        # index = next((i for i, owner in enumerate(owners_list) if f"{owner.first_name} {owner.last_name}" == owner_full_name), -1)
        #
        # if index != -1:
        #     owner_information = {
        #         "owner_id": owners_list[index].id,
        #         "owner_first_name": owners_list[index].first_name,
        #         "owner_last_name": owners_list[index].last_name,
        #         "owner_email": owners_list[index].email
        #     }
        #
        #     pprint(owners_response)
        #     return str(owner_information)
        # else:
        #     return OWNER_NOT_FOUND_ERROR
    except ApiException as e:
        logger.info("Exception when calling owner_api: %s\n" % e)
