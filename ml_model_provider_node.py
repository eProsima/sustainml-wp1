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
"""SustainML ML Model Provider Node Implementation."""

from sustainml_py.nodes.MLModelNode import MLModelNode

# Manage signaling
import os
import signal
import threading
import time
import json

from rdftool.ModelONNXCodebase import model
from rdftool.rdfCode import load_graph, get_models_for_problem, get_problems, get_model_details, print_models

# Whether to go on spinning or interrupt
running = False

# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    MLModelNode.terminate()
    global running
    running = False

# User Callback implementation
# Inputs: ml_model_metadata, app_requirements, hw_constraints, ml_model_baseline, hw_baseline, carbonfootprint_baseline
# Outputs: node_status, ml_model
def task_callback(ml_model_metadata,
                  app_requirements,
                  hw_constraints,
                  ml_model_baseline,
                  hw_baseline,
                  carbonfootprint_baseline,
                  node_status,
                  ml_model):

    # Callback implementation here

    print (f"Received Task: {ml_model_metadata.task_id().problem_id()},{ml_model_metadata.task_id().iteration_id()}")

    if not ml_model_metadata.ml_model_metadata().empty():

        try:

            graph = load_graph(os.path.dirname(__file__)+'/graph_v2.ttl')
            metadata = ml_model_metadata.ml_model_metadata()[0]

            # Model selection and information retrieval
            suggested_models = get_models_for_problem(graph, metadata)
            # print("Suggested models ")
            print_models(suggested_models)
            # model_info = get_model_details(graph, suggested_models)   # WIP - use for model information
            # model_names = [info['name'] for info in model_info]
            model_names = [model[0] for model in suggested_models]

            # Random Model is selected here. In the Final code there should be some sort of selection to choose between Possible Models
            chosen_model = model_names[1]
            print(f"")    #Debugging
            print(f"Chosen model: {chosen_model}")    #Debugging

            # Generate model code and keywords
            onnx_path = model(chosen_model)     # WIP - Further development needed
            ml_model.model(chosen_model)
            ml_model.model_path(onnx_path)

        except Exception as e:
            print(f"Error providing valid MLModel: {e}")
            return
    else:
        raise Exception(f"Failed to determine ML goal for task {ml_model_metadata.task_id()}.")

# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here

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
    node = MLModelNode(callback=task_callback, service_callback=configuration_callback)
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
