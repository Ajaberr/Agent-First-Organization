import json
import shopify
import logging

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifySlots, ShopifyOutputs

from arklex.env.tools.tools import register_tool

logger = logging.getLogger(__name__)

description = "Cancel order by order id."
slots = [
    ShopifySlots.CANCEL_ORDER_ID,
    *PAGEINFO_SLOTS
]

outputs = [
    ShopifyOutputs.CANECEL_REQUEST_DETAILS,
]
ORDER_CANCEL_ERROR = "error: order cancel failed"

errors = [
    ORDER_CANCEL_ERROR,
]

@register_tool(description, slots, outputs, lambda x: x[0] not in errors)
def cancel_order(cancel_order_id: str, **kwargs) -> str:
    auth = authorify_admin(kwargs)
    if auth["error"]:
        return auth["error"]
    
    try:
        with shopify.Session.temp(**auth["value"]):
            response = shopify.GraphQL().execute(f"""
            mutation orderCancel {{
            orderCancel(
                orderId: "{cancel_order_id}",
                reason: CUSTOMER,
                notifyCustomer: true,
                restock: true,
                refund: true
            ) {{
                userErrors {{
                    field
                    message
                }}
            }}
            }}
            """)
            try:
                response = json.loads(response)["data"]
                if not response.get("orderCancel", {}).get("userErrors"):
                    return "The order is successfully cancelled. " + json.dumps(response)
                else:
                    return ORDER_CANCEL_ERROR + json.dumps(response["orderCancel"]["userErrors"])
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                return ORDER_CANCEL_ERROR
    
    except Exception as e:
        return ORDER_CANCEL_ERROR