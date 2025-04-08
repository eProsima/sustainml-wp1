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

# Global variable of the graph
graph = None

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
            chosen_model = None
            # Model restriction after various outputs
            extra_data_bytes = ml_model_metadata.extra_data()
            if extra_data_bytes:
                extra_data_str = ''.join(chr(b) for b in extra_data_bytes)
                extra_data_dict = json.loads(extra_data_str)
                if "model_selected" in extra_data_dict:
                    chosen_model = extra_data_dict["model_selected"]
                    print("Selected model:", chosen_model)  ##debugging

            if chosen_model is None:
                metadata = ml_model_metadata.ml_model_metadata()[0]

                # Model selection and information retrieval
                suggested_models = get_models_for_problem(graph, metadata)
                # print("Suggested models ")
                # print_models(suggested_models)
                # model_info = get_model_details(graph, suggested_models)   # WIP - use for model information
                # model_names = [info['name'] for info in model_info]
                model_names = [model[0] for model in suggested_models]

                # Random Model is selected here. In the Final code there should be some sort of selection to choose between Possible Models
                for model_use in model_names:
                    # Some models can't be downloaded from HF, TODO: Works for all models
                    if(str(model_use) == "meta-llama/Llama-3.1-8B-Instruct"):
                        model_use  =  "openai-community/gpt2"
                    if(str(model_use) == "mlx-community/Llama-3.2-1B-Instruct-4bit"):
                        model_use  =  "openai-community/gpt2-medium"
                    chosen_model = model_use
                    break

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

    if 'model_from_goal' in req.configuration():
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())

        try:
            goal = req.configuration()[len("model_from_goal, "):]
            models = get_models_for_problem(graph, goal)

            sorted_models = ', '.join(sorted([str(m[0]) for m in models]))

            if not sorted_models:
                res.success(False)
                res.err_code(1)  # 0: No error || 1: Error
            else:
                res.success(True)
                res.err_code(0)  # 0: No error || 1: Error

            print(f"Models for {goal}: {sorted_models}")
            res.configuration(json.dumps(dict(models=sorted_models)))

        except Exception as e:
            print(f"Error getting model from goal from request: {e}")
            res.success(False)
            res.err_code(1)

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
    node = MLModelNode(callback=task_callback, service_callback=configuration_callback)
    global graph
    graph = load_graph(os.path.dirname(__file__)+'/graph_v2.ttl')
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
