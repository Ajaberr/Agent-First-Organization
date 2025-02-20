import logging

from langgraph.graph import StateGraph, START

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.env.workers.utils.chat_client import ChatClient

from arklex.orchestrator.NLU.nlu import SlotFilling

logger = logging.getLogger(__name__)
# @register_worker
class HITLWorker(BaseWorker):  
    description = "This is a template for a HITL worker."
    mode = None
    params = None
    verifier = []
    
    slot_fill_api: SlotFilling = None
    
    # def __init__(self, server_ip: str, server_port: int, name: str):
    def __init__(self, **kwargs):
        super().__init__()
                
        self.action_graph = self._create_action_graph()

    def verify_literal(self, message: str) -> bool:
        ''' Override this method to allow verification on just the message with slotfilling'''
        return True
    
    def verify_slots(self, message) -> bool:
        ''' Override this method to allow verification on the slots'''
        return True
    
    def verify(self, state: MessageState) -> bool:
        ''' Override this method to allow advanced verification on MessageState object'''
        if not self.verify_literal(state['user_message'].message):
            return False
        
        if not self.verify_slots(state['slots']):
            return False
        
        return True
    
    def initialize_slotfillapi(self, slotsfillapi):
        self.slotfillapi = SlotFilling(slotsfillapi)
        
    def create_prompt(self):
        ''' Create a prompt for the HITL mc worker '''
        return self.params['intro'] + '\n' + '\n'.join(f"({key}) {item}" for key, item in self.params['choices'].items())
            
    def chat(self, state: MessageState) -> MessageState:
        ''' Connects to chat with the human in the loop '''
        client = ChatClient(self.server_ip, self.server_port, name=self.name, mode='c')
        return client.sync_main()
        
        # arklex pseudocode
        # chat_history = await server_chat() # BACKEND CHATS WITH USER HERE'''
        # state['messageFlow'] = to_message_flow(chat_history)
        # messageFlow['result'] = chat_history[-1]
        return state
    
    def multiple_choice(self, state: MessageState) -> MessageState:
        ''' Connects to give human in the loop multiple choice'''
        client = ChatClient(self.server_ip, self.server_port, name=self.name, mode='ro')
        return client.sync_main(message=self.create_prompt())
    
    def hitl(self, state: MessageState) -> str:
        ''' Human in the loop function '''
        result = None
        match self.mode:
            case "chat":
                chat_result = self.chat(state)
                state['user_message'].history += ('\n' + chat_result)
                state['user_message'].message = chat_result.split(f'{self.name}: ')[-1].split(':')[0]
                result = "Live Chat Completed"

            case "mc":
                attempts = self.params["max_retries"]
                
                for _ in range(attempts):
                    selection = self.multiple_choice(state)

                    result = self.params["choices"].get(selection)
                        
                    if result:
                        break
                else:
                    result = self.params['default']
            
            case _:
                return self.error(state)
                
        state['response'] = result
        return state
        
    def error(self, state: MessageState) -> str:
        ''' Error function '''
        state['response'] = "An Error occurred, please try again"
        state["isComplete"] = False
        return state
            
    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("hitl", self.hitl)
        # Add edges
        workflow.add_edge(START, "hitl")
        return workflow
        
    def execute(self, state: MessageState) -> MessageState:
        if not self.verify(state):
            return self.error(state)
        
        graph = self.action_graph.compile()
        result = graph.invoke(state)
        return result
    
@register_worker
class HITLWorkerTestChat(HITLWorker):
    description = "Chat with a real end user"
    mode = "chat"
    
    def __init__(self, **kwargs):
        super().__init__()
        self.server_ip = kwargs.get('server_ip')
        self.server_port = kwargs.get('server_port')
        self.name = kwargs.get('name')
        
        if not self.server_ip or not self.server_port:
            raise ValueError("Server IP and Port are required")
    
    def verify_literal(self, message: str) -> bool:
        return "chat" in message
    
@register_worker
class HITLWorkerTestMC(HITLWorker):
    description = "Get confirmation from a real end user in purchasing"
    mode = "mc"
    params = {
        "intro": "Should the user continue with this purchase? (Y/N)",
        "max_retries": 5,
        'default': 'User is not allowed to continue with the purchase',
        "choices": {
            "Y": "User is allowed to continue with the purchase",
            "N": "User is not allowed to continue with the purchase"
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        self.server_ip = kwargs.get('server_ip')
        self.server_port = kwargs.get('server_port')
        self.name = kwargs.get('name')
    
    def verify_literal(self, message: str) -> bool:
        return "buy" in message
    
@register_worker
class HITLWorkerChatFlag(HITLWorker):
    description = "Human in the loop worker"
    mode = "chat"
    
    def verify_literal(self, message: str) -> bool:
        return "live" in message
    
    def execute(self, state: MessageState) -> MessageState:
        if not state['metadata'].get('hitl'):
            if not self.verify(state):
                return self.error(state)
            state['response'] = '[[Connecting to live chat : this should not show up for user]]'
            state['metadata']['hitl'] = 'mc'
            state['status'] = StatusEnum.INCOMPLETE.value
        
        else:
            state['response'] = 'Live chat completed'
            state['metadata']['hitl'] = None
            state['status'] = StatusEnum.COMPLETE.value
        
        logger.info(state['response'])
        return state
    
@register_worker
class HITLWorkerMCFlag(HITLWorker):
    description = "Get confirmation from a real end user in purchasing"
    mode = "mc"
    params = {
        "intro": "Should the user continue with this purchase? (Y/N)",
        "max_retries": 5,
        'default': 'User is not allowed to continue with the purchase',
        "choices": {
            "Y": "User is allowed to continue with the purchase",
            "N": "User is not allowed to continue with the purchase"
        }
    }
    
    def verify_literal(self, message: str) -> bool:
        return "buy" in message
    
    def execute(self, state: MessageState) -> MessageState:
        if not state['metadata'].get('hitl'):
            if not self.verify(state):
                return self.error(state)
            state['response'] = '[[sending confirmation : this should not show up for user]]'
            state['metadata']['hitl'] = 'mc'
            state['metadata]']['attempts'] = self.params.get("max_retries", 3)
            state['status'] = StatusEnum.INCOMPLETE.value
        
        else:
            result = self.params["choices"].get(state.user_message.message) # not actually user message but system confirmation
            
            if result:
                state['response'] = result
                state['metadata']['hitl'] = None
                state['status'] = StatusEnum.COMPLETE.value
                return state
            
            state['metadata]']['attempts'] -= 1
            if state['metadata]']['attempts'] <= 0:
                state['response'] = self.params['default']
                state['metadata']['hitl'] = None
                state['status'] = StatusEnum.COMPLETE.value
                return state
                    
            state['response'] = '[[sending confirmation : this should not show up for user]]'
            state['metadata']['hitl'] = 'mc'
            state['status'] = StatusEnum.INCOMPLETE.value
            
        return state