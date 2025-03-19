from arklex.env.tools.shopify.utils_slots import ShopifySlots, ShopifyOutputs
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.tools import register_tool

import logging

logger = logging.getLogger(__name__)

description = "Get cart information"
slots = [
    ShopifySlots.CART_ID,
    *PAGEINFO_SLOTS
]
outputs = [
    ShopifyOutputs.GET_CART_DETAILS,
    *PAGEINFO_OUTPUTS
]
CART_NOT_FOUND_ERROR = "error: cart not found"
errors = [CART_NOT_FOUND_ERROR]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_cart(cart_id, **kwargs):
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth = authorify_storefront(kwargs)
    if auth["error"]:
        return auth["error"]

    variable = {
        "id": cart_id,
    }
    headers = {
        "X-Shopify-Storefront-Access-Token": auth["storefront_token"]
    }
    query = f'''
        query ($id: ID!) {{ 
            cart(id: $id) {{
                id
                checkoutUrl
                lines ({nav[0]}) {{
                    nodes {{
                        id
                        quantity
                        merchandise {{
                            ... on ProductVariant {{
                                id
                                title
                                product {{
                                    title
                                    id
                                }}
                            }}
                        }}
                    }}
                    pageInfo {{
                        endCursor
                        hasNextPage
                        hasPreviousPage
                        startCursor
                    }}
                }}
            }}
        }}
    '''
    response = requests.post(auth["storefront_url"], json={'query': query, 'variables': variable}, headers=headers)
    if response.status_code == 200:
        response = response.json()
        cart_data = response["data"]["cart"]
        if not cart_data:
            return CART_NOT_FOUND_ERROR
        response_text = ""
        response_text += f"Checkout URL: {cart_data['checkoutUrl']}\n"
        lines = cart_data['lines']
        for line in lines['nodes']:
            product = line.get("merchandise", {}).get("product", {})
            if product:
                response_text += f"Product ID: {product['id']}\n"
                response_text += f"Product Title: {product['title']}\n"
        return response_text
    else:
        logger.error(f"Error: {response.text}")
        return CART_NOT_FOUND_ERROR