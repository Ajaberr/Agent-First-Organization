# Go to the parent folder of this file (calendar), then Run python -m unittest test.py to test the code in this file.
import argparse
import logging
import unittest
import json
import requests
import base64
import sys
import uuid
import os

# May not need after pip install agentorg
sys.path.insert(0, os.path.abspath('../../'))
print(sys.path)

from arklex.orchestrator.orchestrator import AgentOrg
from arklex.env.env import Env
from arklex.env.tools.tools import logger


class Logic_Test(unittest.TestCase):
    file_path = "test_cases.json"
    with open(file_path, "r", encoding="UTF-8") as f:
        TEST_CASES = json.load(f)

    @classmethod
    def setUpClass(cls):
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        cls.user_prefix = "user"
        cls.worker_prefix = "assistant"
        file_path = "taskgraph.json"
        with open(file_path, "r", encoding="UTF-8") as f:
            cls.config = json.load(f)
        cls.env = Env(
            tools=cls.config.get("tools", []),
            workers=cls.config.get("workers", []),
            slotsfillapi=cls.config["slotfillapi"]
        )
        cls.config["input_dir"] = './'
        os.environ["DATA_DIR"] = cls.config["input_dir"]

    @classmethod
    def tearDownClass(cls):
        """Method to tear down the test fixture. Run AFTER the test methods."""
        pass

    def _get_api_bot_response(self, user_text, history, params):

        data = {"text": user_text, 'chat_history': history, 'parameters': params}
        orchestrator = AgentOrg(config=self.config, env=self.env)
        result = orchestrator.get_response(data)

        return result['answer'], result['parameters']

    def test_Unittest0(self):
        logger.info("\n=============Unit Test 0=============")
        logger.info(f"{self.TEST_CASES[0]['description']}")
        history = []
        params = {}
        nodes = []
        for node in self.config['nodes']:
            if node[1].get("type", "") == 'start':
                start_message = node[1]['attribute']["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[0]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            nodes.append(params.get("curr_node"))
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
        print(f"{self.TEST_CASES[0]['criteria']}")
        # print(json.dumps(history, indent=4))
        self.assertEqual(nodes, self.TEST_CASES[0]["trajectory"])

    def test_Unittest1(self):
        print("\n=============Unit Test 1=============")
        print(f"{self.TEST_CASES[1]['description']}")
        history = []
        params = {}
        nodes = []
        for node in self.config['nodes']:
            if node[1].get("type", "") == 'start':
                start_message = node[1]['attribute']["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[1]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            nodes.append(params.get("curr_node"))
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
        print(f"{self.TEST_CASES[1]['criteria']}")
        # print(json.dumps(history, indent=4))
        self.assertEqual(nodes, self.TEST_CASES[1]["trajectory"])

    def test_Unittest2(self):
        print("\n=============Unit Test 2=============")
        print(f"{self.TEST_CASES[2]['description']}")
        history = []
        params = {}
        nodes = []
        for node in self.config['nodes']:
            if node[1].get("type", "") == 'start':
                start_message = node[1]['attribute']["value"]
                break
        history.append({"role": self.worker_prefix, "content": start_message})

        for user_text in self.TEST_CASES[2]["user_utterance"]:
            print(f"User: {user_text}")
            output, params = self._get_api_bot_response(user_text, history, params)
            print(f"Bot: {output}")
            nodes.append(params.get("curr_node"))
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.worker_prefix, "content": output})
        print(f"{self.TEST_CASES[2]['criteria']}")
        # print(json.dumps(history, indent=4))
        self.assertEqual(nodes, self.TEST_CASES[2]["trajectory"])


if __name__ == '__main__':
    unittest.main()
