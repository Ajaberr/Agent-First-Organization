from typing import Any, Dict
import json

import shopify

from arklex.env.tools.tools import register_tool
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifySlots, ShopifyOutputs

description = "Find user id by email. If the user is not found, the function will return an error message."
slots = [
    ShopifySlots.USER_EMAIL
]
outputs = [
    ShopifyOutputs.USER_ID
]

USER_NOT_FOUND_ERROR = "error: user not found"
MULTIPLE_USERS_SAME_EMAIL_ERROR = "error: there are multiple users with the same email"
errors = [
    USER_NOT_FOUND_ERROR,
    MULTIPLE_USERS_SAME_EMAIL_ERROR
]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def find_user_id_by_email(user_email: str, **kwargs) -> str:
    auth = authorify_admin(kwargs)
    if auth["error"]:
        return auth["error"]
    
    user_id = ""
    
    try:
        with shopify.Session.temp(**auth["value"]):
            response = shopify.GraphQL().execute(f"""
                {{
                    customers (first: 10, query: "email:{user_email}") {{
                        edges {{
                            node {{
                                id
                            }}
                        }}
                    }}
                }}
                """)
        nodes = json.loads(response)["data"]["customers"]["edges"]
        if len(nodes) == 1:
            user_id = nodes[0]["node"]["id"]
            return user_id
        else:
            return MULTIPLE_USERS_SAME_EMAIL_ERROR
    except Exception as e:
        return USER_NOT_FOUND_ERROR