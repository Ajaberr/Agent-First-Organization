import ast
from datetime import datetime

import hubspot
from hubspot.crm.objects.emails import ApiException
from hubspot.crm.associations.v4 import AssociationSpec
from hubspot.crm.tickets.models import SimplePublicObjectInputForCreate

from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.hubspot.utils import HUBSPOT_AUTH_ERROR


description = "Create a ticket for the existing customer when the customer has some problem about the specific product."


slots = [
    {
        "name": "cus_cid",
        "type": "string",
        "description": "The id of the customer contact.",
        "prompt": "",
        "required": True,
    },
    {
        "name": "issue",
        "type": "string",
        "description": "The question that the customer has for the specific product",
        "prompt": "",
        "required": True,
    }
]
outputs = [
    {
        "name": "ticket_id",
        "type": "string",
        "description": "The id of the ticket for the existing customer and the specific issue",
    }
]

USER_NOT_FOUND_ERROR = "error: user not found (not an existing customer)"
TICKET_CREATION_ERROR = "error: ticket creation failed"
errors = [
    HUBSPOT_AUTH_ERROR,
    TICKET_CREATION_ERROR
]


@register_tool(description, slots, outputs, lambda x: x not in errors)
def create_ticket(cus_cid: str, issue: str, **kwargs) -> str:
    access_token = kwargs.get('access_token')
    if not access_token:
        return HUBSPOT_AUTH_ERROR

    api_client = hubspot.Client.create(access_token=access_token)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z"
    subject_name = "Issue of " + cus_cid + " at " + timestamp
    ticket_properties = {
        'hs_pipeline_stage': 1,
        'content': issue,
        'subject': subject_name
    }
    ticket_for_create = SimplePublicObjectInputForCreate(properties=ticket_properties)
    try:
        ticket_creation_response = api_client.crm.tickets.basic_api.create(simple_public_object_input_for_create=ticket_for_create)
        ticket_creation_response = ticket_creation_response.to_dict()
        ticket_id = ticket_creation_response['id']
        association_spec = [
            AssociationSpec(
                association_category="HUBSPOT_DEFINED",
                association_type_id=15
            )
        ]
        try:
            association_creation_response = api_client.crm.associations.v4.basic_api.create(
                object_type="contact",
                object_id=cus_cid,
                to_object_type="ticket",
                to_object_id=ticket_id,
                association_spec=association_spec
            )
            return ticket_id
        except ApiException as e:
            logger.info("Exception when calling AssociationV4: %s\n" % e)
            return TICKET_CREATION_ERROR
    except ApiException as e:
        logger.info("Exception when calling Crm.tickets.create: %s\n" % e)
        return TICKET_CREATION_ERROR





