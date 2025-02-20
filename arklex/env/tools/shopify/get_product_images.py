import json
import logging

import shopify

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify

# ADMIN
from arklex.env.tools.shopify.utils_slots import ShopifySlots, ShopifyOutputs
from arklex.env.tools.tools import register_tool

logger = logging.getLogger(__name__)

description = "Get the product image url of a product."
slots = [
    ShopifySlots.PRODUCT_IDS,
    *PAGEINFO_SLOTS
]
outputs = [
    ShopifyOutputs.PRODUCTS_DETAILS,
    *PAGEINFO_OUTPUTS
]
PRODUCTS_NOT_FOUND = "error: product not found"
errors = [PRODUCTS_NOT_FOUND]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_product_images(product_ids: list, **kwargs) -> str:
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth = authorify(kwargs)
    if auth["error"]:
        return auth["error"]

    try:
        ids = ' OR '.join(f'id:{pid.split("/")[-1]}' for pid in product_ids)
        with shopify.Session.temp(**auth["value"]):
            response = shopify.GraphQL().execute(f"""
                {{
                    products ({nav[0]}, query:"{ids}") {{
                        nodes {{
                            title
                            description
                            onlineStoreUrl
                            images(first: 1) {{
                                edges {{
                                    node {{
                                        src
                                        altText
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
            """)
            result = json.loads(response)['data']['products']
            response = result["nodes"]
            response_text = ""
            product_list = []
            for i, product in enumerate(response):
                response_text += f"Product {i+1}:\n"
                response_text += f"Product Title: {product.get('title')}\n"
                response_text += f"Product Description: {product.get('description')}\n\n"
                product_dict = {"title": product.get('title'), 
                                "description": product.get('description'), 
                                "product_url": product.get('onlineStoreUrl'),
                                "image_url" : product.get('images', {}).get('edges', [{}])[0].get('node', {}).get('src', ""),
                            }
                product_list.append(product_dict)
            return json.dumps({
                "content": response_text,
                "product_list": product_list
            })
    except Exception as e:
        return PRODUCTS_NOT_FOUND
