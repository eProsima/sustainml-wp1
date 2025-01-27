# Copyright 2023 SustainML Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SustainML Task Encoder Node Implementation."""

from sustainml_py.nodes.MLModelMetadataNode import MLModelMetadataNode

# Manage signaling
import os
import signal
import threading
import time
import json

from rdftool.rdfCode import get_mlgoals, get_inputs
from ollama import Client

# Whether to go on spinning or interrupt
running = False

# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    MLModelMetadataNode.terminate()
    global running
    running = False

def get_llm_response(client, model_version, problem_definition, prompt):
    """Get a response from the Ollama API."""
    prompt = f"Given the following Information: \"{problem_definition}\". {prompt}"
    try:
        response = client.chat(model=model_version, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Error in getting response from Ollama: {e}")
        return None

# User Callback implementation
# Inputs: user_input
# Outputs: node_status, ml_model_metadata
def task_callback(user_input, node_status, ml_model_metadata):

    # Callback implementation here

    print (f"Received Task: {user_input.task_id().problem_id()},{user_input.task_id().iteration_id()}")

    client = Client(host='http://localhost:11434')
    graph_path = os.path.dirname(__file__)+'/CustomGraph.ttl'

    # Retrieve Possible Ml Goals from graph
    mlgoals = get_mlgoals(graph_path)
    goals = ', '.join(mlgoals)

    # Select MLGoal Using Ollama llama 3
    prompt = f"Which of the following machine learning Goals can be used to solve this problem (or part of it): {goals}. Answer with only  one of the Machine learning goals and with nothing else"
    mlgoal = get_llm_response(client, "llama3", user_input.problem_definition(), prompt)

    if mlgoal != None and mlgoal in mlgoals:
        ml_model_metadata.ml_model_metadata().append(mlgoal)
        print (f"Selected ML Goal: {mlgoal}")
    else:
        raise Exception(f"Failed to determine ML goal for task {user_input.task_id()}.")

# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here

    if req.configuration() == "modality":
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())

        # Retrieve Possible Ml Goals from graph
        graph_path = os.path.dirname(__file__)+'/CustomGraph.ttl'
        inputs = get_inputs(graph_path)
        ins = ', '.join(inputs)

        if ins == "":
            res.success(False)
            res.err_code(1) # 0: No error || 1: Error
        else:
            res.success(True)
            res.err_code(0) # 0: No error || 1: Error
        print (f"Available Modalities: {ins}")
        res.configuration(json.dumps(dict(modalities=ins)))

    else:
        # Dummy JSON configuration and implementation
        dummy_config = {
            "param1": "value1",
            "param2": "value2",
            "param3": "value3"
        }
        res.configuration(json.dumps(dummy_config))
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())
        res.success(True)
        res.err_code(0) # 0: No error || 1: Error


# Main workflow routine
def run():
    node = MLModelMetadataNode(callback=task_callback, service_callback=configuration_callback)
    global running
    running = True
    node.spin()

# Call main in program execution
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    """Python does not process signals async if
    the main thread is blocked (spin()) so, tun
    user work flow in another thread """
    runner = threading.Thread(target=run)
    runner.start()

    while running:
        time.sleep(1)

    runner.join()
