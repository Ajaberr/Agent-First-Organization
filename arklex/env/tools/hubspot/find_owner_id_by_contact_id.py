import json
from datetime import datetime, timezone
from ..tools import register_tool, logger
import hubspot
from hubspot.crm.objects.emails import PublicObjectSearchRequest, ApiException
from hubspot.crm.objects.communications.models import SimplePublicObjectInputForCreate
from hubspot.crm.associations.v4 import AssociationSpec
from arklex.env.tools.hubspot.utils import HUBSPOT_AUTH_ERROR
from pprint import pprint
import ast
description = "Find the owner id in the contact. If owner id is found, the next step is using the extracted owner id to find the information of the owner. "


slots = [
    {
        "name": "customer_contact_information",
        "type": "string",
        "description": "After finding the exiting customer, the detailed information of the customer (actually a dict) is provided",
        "prompt": "",
        "required": True,
    }
]

outputs = [
    {
        "name": "owner_id",
        "type": "string",
        "description": "The id of the owner of the contact.",
    }
]

errors = [
    HUBSPOT_AUTH_ERROR
]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def find_owner_id_by_contact_id(customer_contact_information, **kwargs) -> str:
    access_token = kwargs.get('access_token')
    if not access_token:
        return HUBSPOT_AUTH_ERROR

    customer_contact_information = ast.literal_eval(customer_contact_information)
    customer_id = customer_contact_information.get('contact_id')
    api_client = hubspot.Client.create(access_token=access_token)

    try:
        get_owner_id_response = api_client.api_request(
            {
                "path": "/crm/v3/objects/contacts/{}".format(customer_id),
                "method": "GET",
                "headers": {
                    'Content-Type': 'application/json'
                },
                "qs": {
                    "properties": 'hubspot_owner_id'
                }
            }

        )
        get_owner_id_response = get_owner_id_response.json()

        owner_id = get_owner_id_response['properties']['hubspot_owner_id']

        return str(owner_id)
    except ApiException as e:
        logger.info("Exception when extracting owner_id of one contact: %s\n" % e)
