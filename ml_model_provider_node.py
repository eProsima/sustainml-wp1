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
from rdftool.rdfCode import load_graph, get_models_for_problem, get_models_for_problem_and_tag, get_problems, get_model_details, print_models

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

    try:
        chosen_model = None
        # Model restriction after various outputs
        restrained_models = []
        type = None
        extra_data_bytes = ml_model_metadata.extra_data()
        if extra_data_bytes:
            extra_data_str = ''.join(chr(b) for b in extra_data_bytes)
            extra_data_dict = json.loads(extra_data_str)
            if "type" in extra_data_dict:
                type = extra_data_dict["type"]

            if "model_restrains" in extra_data_dict:
                restrained_models = extra_data_dict["model_restrains"]

            if "model_selected" in extra_data_dict:
                chosen_model = extra_data_dict["model_selected"]
                print("Model already selected: ", chosen_model)

        if chosen_model is None:
            metadata = ml_model_metadata.ml_model_metadata()[0]

            # Model selection and information retrieval
            global graph
            if type is not None:
                print(f"Limiting search to models with tag: {type}")
                suggested_models = get_models_for_problem_and_tag(graph, metadata, type)
            else:
                suggested_models = get_models_for_problem(graph, metadata)

            # model_info = get_model_details(graph, suggested_models)
            # model_names = [info['name'] for info in model_info]
            model_names = [model[0] for model in suggested_models]

            # Random Model is selected here. In the Final code there should be some sort of selection to choose between Possible Models
            for model_use in model_names:
                # Some models can't be downloaded from HF, TODO: Works for all models
                if "llama" in str(model_use).lower():
                    continue
                if str(model_use) not in restrained_models:
                    chosen_model = model_use
                    break
                else:
                    print(f"Chosen model: {model_use} is restrained. The restrained models are {restrained_models}. Choosing the next model.")
            else:
                raise Exception("No valid model could be selected.")

        print(f"ML Model chosen: {chosen_model}")

        # Generate model code and keywords
        onnx_path = model(chosen_model)     # TODO - Further development needed
        ml_model.model(chosen_model)
        ml_model.model_path(onnx_path)

    except Exception as e:
        print(f"Failed to determine ML model for task {ml_model_metadata.task_id()}: {e}.")
        ml_model.model("Error")
        ml_model.model_path("Error")
        error_message = "Failed to obtain ML model for task: " + str(e)
        error_info = {"error": error_message}
        encoded_error = json.dumps(error_info).encode("utf-8")
        ml_model.extra_data(encoded_error)

# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here
    global graph
    if 'model_from_goal' in req.configuration():
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())

        try:
            text = req.configuration()[len("model_from_goal, "):]
            parts = text.split(',')
            if len(parts) >= 2:
                goal = parts[0].strip()
                tag = parts[1].strip()
                models = get_models_for_problem_and_tag(graph, goal, tag)
            else:
                goal = text.strip()
                models = get_models_for_problem(graph, goal)

            sorted_models = ', '.join(sorted([str(m[0]) for m in models]))

            if not sorted_models:
                res.success(False)
                res.err_code(1)  # 0: No error || 1: Error
            else:
                res.success(True)
                res.err_code(0)  # 0: No error || 1: Error

            print(f"Models for {goal}: {sorted_models}")    #debug
            res.configuration(json.dumps(dict(models=sorted_models)))

        except Exception as e:
            print(f"Error getting model from goal from request: {e}")
            res.success(False)
            res.err_code(1)

    else:
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())
        error_msg = f"Unsupported configuration request: {req.configuration()}"
        res.configuration(json.dumps({"error": error_msg}))
        res.success(False)
        res.err_code(1) # 0: No error || 1: Error
        print(error_msg)


# Main workflow routine
def run():
    global graph
    graph = load_graph(os.path.dirname(__file__)+'/graph_v2.ttl')
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
