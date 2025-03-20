from arklex.env.tools.shopify.utils_slots import ShopifyCartAddItemsSlots, ShopifyOutputs
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *

from arklex.env.tools.tools import register_tool

description = "Add items to user's shopping cart."
slots = ShopifyCartAddItemsSlots.get_all_slots()
outputs = [
    ShopifyOutputs.CART_ADD_ITEMS_DETAILS
]
CART_ADD_ITEM_ERROR = "error: products could not be added to cart"
errors = [CART_ADD_ITEM_ERROR]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def cart_add_items(cart_id: str, add_line_items: list, **kwargs):
    auth = authorify_storefront(kwargs)
    if auth["error"]:
        return auth["error"]
    
    variable = {
        "cartId": cart_id,
        "lines": [
            {"merchandiseId": item,
                "quantity": 1
            } for item in add_line_items
        ]
    }
    headers = {
        "X-Shopify-Storefront-Access-Token": auth["storefront_token"]
    }
    query = '''
    mutation cartLinesAdd($cartId: ID!, $lines: [CartLineInput!]!) {
        cartLinesAdd(cartId: $cartId, lines: $lines) {
            cart {
                checkoutUrl
            }
        }
    }
    '''
    response = requests.post(auth["storefront_url"], json={"query": query, "variables": variable}, headers=headers)
    if response.status_code == 200:
        cart_data = response.json()
        if "errors" in cart_data:
            return CART_ADD_ITEM_ERROR
        else:
            return "Items are successfully added to the shopping cart. " + json.dumps(cart_data["data"]["cartLinesAdd"]["cart"])
    else:
        return CART_ADD_ITEM_ERROR
