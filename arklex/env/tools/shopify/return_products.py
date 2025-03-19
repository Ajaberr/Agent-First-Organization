import json
import shopify
import logging

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifySlots, ShopifyOutputs

from arklex.env.tools.tools import register_tool

logger = logging.getLogger(__name__)

description = "Return order by order id. If no fulfillments are found, the function will return an error message."
slots = [
    ShopifySlots.RETURN_ORDER_ID,
    *PAGEINFO_SLOTS
]
# change output
outputs = [
    ShopifyOutputs.RETURN_REQUEST_DETAILS,
]
PRODUCT_RETURN_ERROR = "error: product return failed"
NO_FULFILLMENT_FOUND_ERROR = "error: no returnable fulfillment found"

errors = [
    PRODUCT_RETURN_ERROR,
    NO_FULFILLMENT_FOUND_ERROR
]

@register_tool(description, slots, outputs, lambda x: x[0] not in errors)
def return_products(return_order_id: str, **kwargs) -> str:
    auth = authorify_admin(kwargs)
    if auth["error"]:
        return auth["error"]
    
    try:
        with shopify.Session.temp(**auth["value"]):
            response = shopify.GraphQL().execute(f"""
            {{
                returnableFulfillments (orderId: "{return_order_id}", first: 10) {{
                    edges {{
                        node {{
                            id
                            fulfillment {{
                                id
                            }}
                            returnableFulfillmentLineItems(first: 10) {{
                                edges {{
                                    node {{
                                        fulfillmentLineItem {{
                                            id
                                        }}
                                        quantity
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            """)
            try:
                response = json.loads(response)
                # Extract all fulfillment line item IDs
                fulfillment_items = []
                for fulfillment in response['data']['returnableFulfillments']['edges']:
                    for line_item in fulfillment['node']['returnableFulfillmentLineItems']['edges']:
                        line_item_id = line_item['node']['fulfillmentLineItem']['id']
                        line_item_quantity = line_item['node']['quantity']
                        fulfillment_items.append({"fulfillmentLineItemId": line_item_id, "quantity": line_item_quantity})
                if not fulfillment_items:
                    return NO_FULFILLMENT_FOUND_ERROR
                logger.info(f"Found {len(fulfillment_items)} fulfillment items.")
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                return NO_FULFILLMENT_FOUND_ERROR

            # Submit the return request
            fulfillment_string = ""
            for item in fulfillment_items:
                fulfillment_string += f"{{fulfillmentLineItemId: \"{item['fulfillmentLineItemId']}\", quantity: {item['quantity']}, returnReason: UNKNOWN}},"
            fulfillment_string = "[" + fulfillment_string + "]"
            response = shopify.GraphQL().execute(f"""
            mutation ReturnRequestMutation {{
            returnRequest(
                input: {{
                orderId: "{return_order_id}",
                returnLineItems: {fulfillment_string}
                }}
            ) {{
                return {{
                    id
                    status
                }}
                userErrors {{
                    field
                    message
                }}
            }}
            }}
            """)
            try:
                response = json.loads(response)["data"]
                if response.get("returnRequest"):
                    return "The product return request is successfully submitted. " + json.dumps(response)
                else:
                    return PRODUCT_RETURN_ERROR + json.dumps(response)
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                return PRODUCT_RETURN_ERROR
    
    except Exception as e:
        return PRODUCT_RETURN_ERROR