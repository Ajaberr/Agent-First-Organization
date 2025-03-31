import copy
import logging
import collections
from typing import Tuple

import networkx as nx
import numpy as np
from langchain_openai import ChatOpenAI


from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.utils import normalize, str_similarity, format_chat_history
from arklex.utils.graph_state import NodeInfo, Params, StatusEnum
from arklex.orchestrator.NLU.nlu import NLU, SlotFilling
from arklex.utils.model_config import MODEL

logger = logging.getLogger(__name__)

class TaskGraphBase:
    def __init__(self, name, product_kwargs):
        self.graph = nx.DiGraph(name=name)
        self.product_kwargs = product_kwargs
        self.create_graph()
        self.intents = self.get_pred_intents() # global intents
        self.start_node = self.get_start_node()

    def create_graph(self):
        raise NotImplementedError

    def get_pred_intents(self):
        intents = collections.defaultdict(list)
        for edge in self.graph.edges.data():
            if edge[2].get("attribute", {}).get("pred", False):
                edge_info = copy.deepcopy(edge[2])
                edge_info["source_node"] = edge[0]
                edge_info["target_node"] = edge[1]
                intents[edge[2].get("intent")].append(edge_info)
        return intents
    
    def get_start_node(self):
        for node in self.graph.nodes.data():
            if node[1].get("type", "") == "start":
                return node[0]
        return None


class TaskGraph(TaskGraphBase):
    def __init__(self, name: str, product_kwargs: dict):
        super().__init__(name, product_kwargs)
        self.unsure_intent = {
                "intent": "others",
                "source_node": None,
                "target_node": None,
                "attribute": {
                    "weight": 1,
                    "pred": False,
                    "definition": "",
                    "sample_utterances": []
                }
            }
        self.initial_node = self.get_initial_flow()
        self.model = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        self.nluapi = NLU(self.product_kwargs.get("nluapi"))
        self.slotfillapi = SlotFilling(self.product_kwargs.get("slotfillapi"))

    def create_graph(self):
        nodes = self.product_kwargs["nodes"]
        edges = self.product_kwargs["edges"]
        # convert the intent into lowercase
        for edge in edges:
            edge[2]['intent'] = edge[2]['intent'].lower()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    def get_initial_flow(self):
        services_nodes = self.product_kwargs.get("services_nodes", None)
        node = None
        if services_nodes:
            candidates_nodes = [v for k, v in services_nodes.items()]
            candidates_nodes_weights = [list(self.graph.in_edges(n, data="attribute"))[0][2]["weight"] for n in candidates_nodes]
            node = np.random.choice(candidates_nodes, p=normalize(candidates_nodes_weights))
        return node

    def jump_to_node(self, pred_intent, intent_idx, params: Params, curr_node):
        logger.info(f"pred_intent in jump_to_node is {pred_intent}")
        try:
            candidates_nodes = [self.intents[pred_intent][intent_idx]]
            candidates_nodes = [node for node in candidates_nodes if params["taskgraph"]["node_limit"].get(node["target_node"], 0) >= 1]
            candidates_nodes_weights = [node["attribute"]["weight"] for node in candidates_nodes]
            if candidates_nodes:
                next_node = np.random.choice([node["target_node"] for node in candidates_nodes], p=normalize(candidates_nodes_weights))
                next_intent = pred_intent
            else:  # This is for protection, logically shouldn't enter this branch
                next_node = curr_node
                next_intent = list(self.graph.in_edges(curr_node, data="intent"))[0][2]
        except Exception as e:
            logger.error(f"Error in jump_to_node: {e}")
            next_node = curr_node
            next_intent = list(self.graph.in_edges(curr_node, data="intent"))[0][2]
        return next_node, next_intent
    
    def move_to_node(self, curr_node, params: Params):
        # if not match other intent, randomly choose one sample from candidate samples
        # if curr_node does not have next connected nodes -> return curr_node
        candidate_samples = []
        candidates_nodes_weights = []
        for out_edge in self.graph.out_edges(curr_node, data=True):
            if out_edge[2]["intent"] == "none" and params["taskgraph"]["node_limit"].get(out_edge[1], 1) >= 1:
                candidate_samples.append(out_edge[1])
                candidates_nodes_weights.append(out_edge[2]["attribute"]["weight"])
        if candidate_samples:
            # randomly choose one sample from candidate samples
            next_node = np.random.choice(candidate_samples, p=normalize(candidates_nodes_weights))
        else:  # leaf node
            next_node = curr_node

        return next_node

    
    def _get_node(self, sample_node, params: Params, intent=None) -> Tuple[NodeInfo, Params]:
        logger.info(f"available_intents in _get_node: {params['taskgraph']['available_global_intents']}")
        logger.info(f"intent in _get_node: {intent}")
        node_info = self.graph.nodes[sample_node]
        resource_name = node_info["resource"]["name"]
        resource_id = node_info["resource"]["id"]
        if intent and params["taskgraph"]["node_limit"].get(sample_node, 1) <= 0 and intent in params["taskgraph"]["available_global_intents"]:
            # delete the corresponding node item from the intent list
            for item in params["taskgraph"]["available_global_intents"].get(intent, []):
                if item["target_node"] == sample_node:
                    params["taskgraph"]["available_global_intents"][intent].remove(item)
            if not params["taskgraph"]["available_global_intents"][intent]:
                params["taskgraph"]["available_global_intents"].pop(intent)
        
        params["taskgraph"]["curr_node"] = sample_node
        
        node_info = NodeInfo(
            resource_id=resource_id,
            resource_name=resource_name,
            attributes=node_info["attribute"],
            add_flow_stack=False
        )
        
        return node_info, params

    def _postprocess_intent(self, pred_intent, available_global_intents):
        found_pred_in_avil = False
        real_intent = pred_intent
        idx = 0
        # check whether there are __<{idx}> in the pred_intent
        if "__<" in pred_intent:
            real_intent = pred_intent.split("__<")[0]
        # get the idx
            idx = int(pred_intent.split("__<")[1].split(">")[0])
        for item in available_global_intents:
            if str_similarity(real_intent, item) > 0.9:
                found_pred_in_avil = True
                real_intent = item
                break
        return found_pred_in_avil, real_intent, idx
    
    # If the local intent is None, determine whether current global intent is finished
    def _switch_pred_intent(self, curr_global_intent, avail_pred_intents):
        if not curr_global_intent:
            return True
        other_pred_intents = [intent for intent in avail_pred_intents.keys() if intent != curr_global_intent and intent != self.unsure_intent.get("intent")]
        logger.info(f"_switch_pred_intent function: curr_global_intent: {curr_global_intent}")
        logger.info(f"_switch_pred_intent function: avail_pred_intents: {other_pred_intents}")

        prompt = f"The assistant is currently working on the task: {curr_global_intent}\nOther available tasks are: {other_pred_intents}\nAccording to the conversation, decide whether the user wants to stop the current task and switch to another one.\nConversation:\n{self.chat_history_str}\nThe response should only be yes or no."
        response = self.model.invoke(prompt)        
        if "no" in response.content.lower():
            return False
        return True
            
    def get_current_node(self, params: Params):
        curr_node = params.get("curr_node", None)
        if not curr_node or curr_node not in self.graph.nodes:
            curr_node = self.start_node
            params["taskgraph"]["curr_node"] = curr_node
            params["taskgraph"]["curr_pred_intent"] = None
        else:
            curr_node = str(curr_node)
        return curr_node, params
    
    def get_available_global_intents(self, params: Params):
        available_global_intents = params["taskgraph"].get("available_global_intents", [])
        if not available_global_intents:
            available_global_intents = copy.deepcopy(self.intents)
            if self.unsure_intent.get("intent") not in available_global_intents.keys():
                available_global_intents[self.unsure_intent.get("intent")].append(self.unsure_intent)
        return available_global_intents
    
    def update_node_limit(self, params: Params):
        old_node_limit = params["taskgraph"].get("node_limit", {})
        node_limit = {}
        for node in self.graph.nodes.data():
            limit = old_node_limit.get(node[0], node[1].get("limit")) 
            if limit is not None:
                node_limit[node[0]] = limit
        params["taskgraph"]["node_limit"] = node_limit
        return params

    def get_local_intent(self, curr_node, params: Params):
        candidates_intents = collections.defaultdict(list)
        for u, v, data in self.graph.out_edges(curr_node, data=True):
            intent = data.get("intent")
            # TODO: do we want to check the limit here or let skip_node to decide only?
            if intent != "none" and data.get("intent") and params["taskgraph"]["node_limit"].get(v, 0) >= 1:
                edge_info = copy.deepcopy(data)
                edge_info["source_node"] = u
                edge_info["target_node"] = v
                candidates_intents[intent].append(edge_info)
        return dict(candidates_intents)
    
    def change_current_global_intent(self, intent, params: Params):
        prev_global_intent = params["taskgraph"].get("curr_pred_intent", None)
        logger.info(f"Global intent changed from {prev_global_intent} to {intent}")
        params["taskgraph"]["curr_pred_intent"] = intent
        return intent, params
    
    def get_last_flow_stack_node(self, params: Params):
        path = params["taskgraph"]["path"]
        for i in range(len(path) - 1, -1, -1):
            if path[i]["in_flow_stack"]:
                params["taskgraph"]["path"][i]["in_flow_stack"] = False
                return params["taskgraph"]["path"][i]["node_id"]
        return None

    def get_node(self, inputs):
        self.text = inputs["text"]
        self.chat_history_str = inputs["chat_history_str"]
        params: Params = inputs["parameters"]
        params["taskgraph"]["nlu_records"] = []

        curr_node, params = self.get_current_node(params)
        logger.info(f"Intial curr_node: {curr_node}")
        node_status = params["taskgraph"].get("node_status", {})
        logger.info(f"node_status: {node_status}")

        # For the multi-step nodes, directly stay at that node instead of moving to other nodes
        status = node_status.get(curr_node, StatusEnum.COMPLETE.value)
        if status == StatusEnum.STAY.value:
            node_info = self.graph.nodes[curr_node]
            node_name = node_info["resource"]["name"]
            id = node_info["resource"]["id"]
            node_output = {"id": id, "name": node_name, "attribute": node_info["attribute"]}
            return node_output, params
        
        if not list(self.graph.successors(curr_node)):  # leaf node
            last_flow_stack_node = self.get_last_flow_stack_node(params)
            if last_flow_stack_node:
                curr_node = last_flow_stack_node
            if self.initial_node:
                curr_node = self.initial_node
        
        # store current node
        params["taskgraph"]["curr_node"] = curr_node
        logger.info(f"curr_node: {curr_node}")
        # get the current global intent
        curr_global_intent = params["taskgraph"].get("curr_pred_intent", None)

        # available global intents
        available_global_intents = self.get_available_global_intents(params)
        logger.info(f"available_global_intents: {available_global_intents}")
        
        # update limit
        params = self.update_node_limit(params)

        # Get local intents of the curr_node
        curr_local_intents = self.get_local_intent(curr_node, params)
        logger.info(f"curr_local_intents: {curr_local_intents}")
        
        
        # next node after curr_node, initially we don't know what is next_node
        next_node = None
        
        # whether has checked global intent or not, since 1 turn only need to check global intent for 1 time
        global_intent_checked = False

        if not curr_local_intents:  # no local intent under the current node
            logger.info(f"no local intent under the current node")
            # if only unsure_intent is available -> move directly to this intent
            if len(available_global_intents) == 1 and self.unsure_intent.get("intent") in available_global_intents.keys():
                pred_intent = self.unsure_intent.get("intent")
            else: # global intent prediction
                # Another checking to make sure the user indeed want to switch the current task
                if not self._switch_pred_intent(curr_global_intent, available_global_intents):
                    logger.info(f"User doesn't want to switch the current task: {curr_global_intent}")
                    pred_intent = self.unsure_intent.get("intent")
                else:
                    logger.info(f"User wants to switch the current task: {curr_global_intent}")
                    global_intent_checked = True
                    # check other intent
                    # if match other intent, add flow, jump over
                    available_global_intents_w_unsure = copy.deepcopy(available_global_intents)
                    available_global_intents_w_unsure[self.unsure_intent.get("intent")] = \
                        available_global_intents_w_unsure.get(self.unsure_intent.get("intent"), [self.unsure_intent])
                    logger.info(f"available_global_intents_w_unsure: {available_global_intents_w_unsure}")
                    
                    pred_intent = self.nluapi.execute(self.text, available_global_intents_w_unsure, self.chat_history_str, params["metadata"])
                    params["taskgraph"]["nlu_records"].append({"candidate_intents": available_global_intents_w_unsure, 
                                        "pred_intent": pred_intent, "no_intent": False, "global_intent": True})
                    found_pred_in_avil, pred_intent, intent_idx = self._postprocess_intent(pred_intent, available_global_intents)
                    if found_pred_in_avil:
                        curr_global_intent, params = self.change_current_global_intent(pred_intent, params)
                        next_node, next_intent = self.jump_to_node(pred_intent, intent_idx, params, curr_node)
                        logger.info(f"curr_node: {next_node}")
                        node_info, params = self._get_node(next_node, params, intent=next_intent)
                        # if current node is not a leaf node and jump to another node, then add it onto stack
                        if next_node != curr_node and list(self.graph.successors(curr_node)):
                            node_info["add_flow_stack"] = True
                        return node_info, params

        
        # if current node is incompleted -> return current node
        status = node_status.get(curr_node, StatusEnum.COMPLETE.value)
        if status == StatusEnum.INCOMPLETE.value:
            logger.info(f"no local or global intent found, the current node is not complete")
            node_info = {
                "id": self.graph.nodes[curr_node]["resource"]["id"],
                "name": self.graph.nodes[curr_node]["resource"]["name"],
                "attribute": self.graph.nodes[curr_node]["attribute"]
            }
            return node_info, params
        
        if not curr_local_intents:
            logger.info(f"no local or global intent found, move to the next connected node(s)")
            next_node = self.move_to_node(curr_node, params)
            if next_node != curr_node:  # continue if not leaf node
                curr_node = next_node
                logger.info(f"curr_node: {curr_node}")
                node_info, params = self._get_node(curr_node, params)
                if params.get("nlu_records", None):
                    params["nlu_records"][-1]["no_intent"] = True  # move on to the next node
                else: # only others available
                    params["nlu_records"] = [{"candidate_intents": [], "pred_intent": "", "no_intent": True, "global_intent": False}]
                return node_info, params

        # handle local intent 111111111111
        # if completed -> randomly choose one of the next connected nodes (edges with intent = None)

        curr_node = params["taskgraph"]["curr_node"]
        next_node = curr_node
        logger.info(f"curr_node: {curr_node}")

        logger.info("Finish global condition, start local intent prediction")
        curr_local_intents_w_unsure = copy.deepcopy(curr_local_intents)
        curr_local_intents_w_unsure[self.unsure_intent.get("intent")] = \
            curr_local_intents_w_unsure.get(self.unsure_intent.get("intent"), [self.unsure_intent])
        logger.info(f"Check intent under current node: {curr_local_intents_w_unsure}")
        pred_intent = self.nluapi.execute(self.text, curr_local_intents_w_unsure, self.chat_history_str, params.get("metadata", {}))
        params["taskgraph"]["nlu_records"].append({"candidate_intents": curr_local_intents_w_unsure, 
                                "pred_intent": pred_intent, "no_intent": False, "global_intent": False})
        found_pred_in_avil, pred_intent, intent_idx = self._postprocess_intent(pred_intent, curr_local_intents)
        logger.info(f"Local intent predition -> found_pred_in_avil: {found_pred_in_avil}, pred_intent: {pred_intent}")
        if found_pred_in_avil:
            if pred_intent.lower() != self.unsure_intent.get("intent") and pred_intent in available_global_intents.keys():
                curr_global_intent, params = self.change_current_global_intent(pred_intent, params)
            for edge in self.graph.out_edges(curr_node, data="intent"):
                if edge[2] == pred_intent:
                    next_node = edge[1]  # found intent under the current node
                    break
            curr_node = next_node
            logger.info(f"curr_node: {curr_node}")
            node_info, params = self._get_node(curr_node, params, intent=pred_intent)
            return node_info, params
        elif not global_intent_checked:  # global intent prediction # TODO: why this?
            # check other intent (including unsure), if found, current flow end, add flow onto stack; if still unsure, then stay at the curr_node, and response without interactive.
            other_intents = collections.defaultdict(list)
            for key, value in available_global_intents.items():
                if key not in curr_local_intents and key in self.intents and key != "none":
                    other_intents[key] = value
            other_intents_w_unsure = copy.deepcopy(other_intents)
            other_intents_w_unsure[self.unsure_intent.get("intent")] = \
                other_intents_w_unsure.get(self.unsure_intent.get("intent"), [self.unsure_intent])
            logger.info(f"Check other intent: {other_intents}")

            pred_intent = self.nluapi.execute(self.text, other_intents_w_unsure, self.chat_history_str, params.get("metadata", {}))
            params["taskgraph"]["nlu_records"].append({"candidate_intents": other_intents, 
                                    "pred_intent": pred_intent, "no_intent": False, "global_intent": True})
            found_pred_in_avil, pred_intent, intent_idx = self._postprocess_intent(pred_intent, other_intents)
            logger.info(f"found_pred_in_avil for global intent: {found_pred_in_avil}, pred_intent: {pred_intent}")
            if found_pred_in_avil:  # found global intent
                if pred_intent.lower() != self.unsure_intent.get("intent"):  # global intent is not unsure
                    curr_global_intent, params = self.change_current_global_intent(pred_intent, params)
                next_node, next_intent = self.jump_to_node(pred_intent, intent_idx, params, curr_node)
                logger.info(f"curr_node: {next_node}")
                node_info, params = self._get_node(next_node, params, intent=next_intent)
                # if current node is not a leaf node and jump to another node, then add it onto stack
                if next_node != curr_node and list(self.graph.successors(curr_node)):
                    node_info["add_flow_stack"] = True
                return node_info, params
            else:
                # If user didn't indicate all the intent of children nodes under the current node, 
                # then we could randomly choose one of Nones to continue the dialog flow
                next_node = self.move_to_node(curr_node, params)
                if next_node != curr_node:  # leaf node or no other nodes to choose from
                    curr_node = next_node
                    logger.info(f"curr_node: {curr_node}")

                    node_info, params = self._get_node(next_node, params)
                    return node_info, params
        # if none of the available intents can represent user's utterance, transfer to the DefaultWorker to let it decide for the next step
        if params["taskgraph"]["nlu_records"]:
            params["taskgraph"]["nlu_records"][-1]["no_intent"] = True  # no intent found
        else:
            params["taskgraph"]["nlu_records"].append({"candidate_intents": [], "pred_intent": "", "no_intent": True, "global_intent": False})
        params["curr_node"] = curr_node
        node_info = NodeInfo(
            resource_id = "default_worker",
            resource_name = "DefaultWorker",
            attributes = {"value": "", "direct": False}
        )
        return node_info, params

    def postprocess_node(self, node) -> Tuple[NodeInfo, Params]:
        node_info: NodeInfo = node[0]
        params: Params = node[1]
        dialog_states = params["taskgraph"].get("dialog_states", {})
        # update the dialog states
        if dialog_states.get(node_info["resource_id"]):
            dialog_states = self.slotfillapi.execute(
                dialog_states.get(node_info["resource_id"]),
                format_chat_history(params["memory"].get("history")),
                params.get("metadata", {})
            )
        params["taskgraph"]["dialog_states"] = dialog_states

        return node_info, params
