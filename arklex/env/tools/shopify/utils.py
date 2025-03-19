import requests
from arklex.env.tools.shopify.auth_utils import get_access_token

SHOPIFY_ADMIN_AUTH_ERROR = "error: missing some or all required shopify admin authentication parameters: shop_url, api_version, admin_token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'admin_token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"
SHOPIFY_STOREFRONT_AUTH_ERROR = "error: missing some or all required shopify storefront authentication parameters: shop_url, api_version, storefront_token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'storefront_token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"

def authorify_admin(kwargs):
    auth = {
        "value": {
            "domain": None,
            "version": None,
            "token": None
        },
        "error": None
    }
    auth["value"]["domain"] = kwargs.get("shop_url")
    auth["value"]["version"] = kwargs.get("api_version")
    auth["value"]["token"] = kwargs.get("admin_token")
    
    if not auth["value"]["domain"] or not auth["value"]["version"] or not auth["value"]["token"]:
        auth["error"] = SHOPIFY_ADMIN_AUTH_ERROR
        return auth
    
    return auth


def authorify_storefront(kwargs):
    auth = {
        "value": {
            "domain": None,
            "version": None,
            "token": None
        },
        "error": None,
        "storefront_token": None,
        "storefront_url": None
    }
    auth["value"]["domain"] = kwargs.get("shop_url")
    auth["value"]["version"] = kwargs.get("api_version")
    auth["value"]["token"] = kwargs.get("storefront_token")
    
    if not auth["value"]["domain"] or not auth["value"]["version"] or not auth["value"]["token"]:
        auth["error"] = SHOPIFY_STOREFRONT_AUTH_ERROR
        return auth
    
    auth["storefront_token"] = auth["value"]["token"]
    auth["storefront_url"] = f"{auth['value']['domain']}/api/{auth['value']['version']}/graphql.json"
    return auth


def make_query(url, query, variables, headers):
    """
    Make query response
    """
    request = requests.post(url, json={'query': query, 'variables': variables}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))
