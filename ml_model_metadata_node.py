# Copyright 2026 SustainML Consortium
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
import signal
import threading
import time
import json
import os
import re
import sys
import pandas as pd

WP2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sustainml-wp2"))
if WP2_ROOT not in sys.path:
    sys.path.insert(0, WP2_ROOT)

import hw_provider_axelera

from rdftool.rdfCode import load_graph, get_problems, get_model_details

from AutoDDG.generate_description import SemanticProfiler, DatasetDescriptionGenerator
from AutoDDG.utils import get_sample, json_to_dataframe
from AutoDDG.data_process import dataset_profiler

from ollama import Client

# Whether to go on spinning or interrupt
running = False

unsupported_goals = [
                "any-to-any",
                "audio-classification",
                "audio-text-to-text",
                "audio-to-audio",
                "automatic-speech-recognition",
                "depth-estimation",
                "document-question-answering",
                "feature-extraction",
                "fill-mask",
                "graph-ml",
                "image-classification",
                "image-feature-extraction",
                "image-segmentation",
                "image-text-to-text",
                "image-to-3d",
                "image-to-image",
                "image-to-text",
                "image-to-video",
                "keypoint-detection",
                "mask-generation",
                "object-detection",
                "question-answering",
                "reinforcement-learning",
                "robotics",
                "sentence-similarity",
                "table-question-answering",
                "tabular-classification",
                "tabular-regression",
                "text-classification",
                "text-ranking",
                "text-to-3d",
                "text-to-audio",
                "text-to-image",
                "text-to-speech",
                "text-to-video",
                "text2text-generation",
                "time-series-forecasting",
                "token-classification",
                "unconditional-image-generation",
                "video-classification",
                "video-text-to-text",
                "video-to-video",
                "visual-document-retrieval",
                "visual-question-answering",
                "voice-activity-detection",
                "zero-shot-classification",
                "zero-shot-image-classification",
                "zero-shot-object-detection"
            ]

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
            }
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Error in getting response from Ollama: {e}")
        return None

# Words a quote must contain to show the need for each criterion
YOLO_CRITERIA_WORDS = {
    "accuracy": r"accura|precis|\bmiss|\berror|reliab|\bexact|safety|medical|quality",
    "speed": r"real[ -]?time|\bfast|\bquick|latency|\bfps\b|frame|\bms\b|millisec|\bspeed|\blive\b|per second",
    "energy": r"batter|\bpower|energy|\bwatt|efficien|sustainab|consum|carbon|\bgreen",
}


def get_yolo_preferences(client, user_input):
    """Read, from the problem description, what matters when choosing among the YOLO models of a
    goal: they all do the same task on the same dataset and only trade accuracy for speed and
    energy. Returns {"priority", "min_fps", "max_latency_ms", "reason"}, "balanced" and no speed
    requirement when the description says nothing about it or the LLM does not answer."""
    text = user_input.problem_short_description()
    if user_input.problem_definition():
        text = f"{text}. {user_input.problem_definition()}"
    if user_input.inputs():
        text = f"{text}. Inputs: {', '.join(user_input.inputs())}"
    if user_input.outputs():
        text = f"{text}. Outputs: {', '.join(user_input.outputs())}"
    if not text.strip(" ."):
        return hw_provider_axelera.normalize_preferences(None)

    # One yes/no per criterion, each backed by the exact words of the problem: a small LLM asked
    # to pick one priority tends to pick the first one offered, and to justify it with words that
    # are not in the problem
    prompt = (
        f"Problem: \"{text}\".\n"
        "Answer only from the words of the problem above, with a JSON object with these keys:\n"
        "\"accuracy\": {\"needed\": true or false, \"quote\": \"\"} - true only if the problem says that "
        "precision or missing nothing is critical;\n"
        "\"speed\": {\"needed\": true or false, \"quote\": \"\"} - true only if the problem asks for real "
        "time, low latency, a frame rate or a time limit;\n"
        "\"energy\": {\"needed\": true or false, \"quote\": \"\"} - true only if the problem mentions "
        "battery, low power or energy;\n"
        "\"min_fps\": the total frames per second to process over all streams if the problem states a "
        "frame rate (e.g. 4 cameras at 30 FPS is 120), else null;\n"
        "\"max_latency_ms\": the maximum time per image in milliseconds if the problem states one, else null.\n"
        "Each quote must copy the exact words of the problem that show the need, or be empty when "
        "needed is false. Most problems state none of these: then all are false."
    )
    try:
        # Temperature 0: every iteration of the problem reads the same requirements, so the
        # next outputs follow one order of models
        response = client.chat(model="llama3", format="json", options={"temperature": 0},
                               messages=[{"role": "user", "content": prompt}])
        answer = json.loads(response["message"]["content"])
    except Exception as e:
        print(f"[ML_MODEL_METADATA] Could not read the YOLO requirements from the description: {e}")
        return hw_provider_axelera.normalize_preferences(None)

    lowered = " ".join(text.lower().split())
    needed = []
    for criterion in ("accuracy", "speed", "energy"):
        item = answer.get(criterion) if isinstance(answer.get(criterion), dict) else {}
        quote = " ".join(str(item.get("quote") or "").lower().split()).strip(" .\"'")
        # The quote must be in the problem and talk about that criterion: the LLM also marks
        # needs with the words of the task itself ("detect objects in images")
        if item.get("needed") is True and quote and quote in lowered and \
                re.search(YOLO_CRITERIA_WORDS[criterion], quote):
            needed.append((criterion, quote))

    prefs = {"priority": needed[0][0] if len(needed) == 1 else "balanced",
             "reason": "; ".join(f"{c}: \"{q}\"" for c, q in needed) or "no need stated in the description"}
    # Numbers only when the problem talks about a frame rate or a time
    if re.search(r"fps|frames? per second|frame rate|hz\b", lowered):
        prefs["min_fps"] = answer.get("min_fps")
    if re.search(r"\d\s*(ms|millisecond|msec|s\b|sec)", lowered):
        prefs["max_latency_ms"] = answer.get("max_latency_ms")
    prefs = hw_provider_axelera.normalize_preferences(prefs)
    print(f"[ML_MODEL_METADATA] YOLO requirements from the description: {prefs}")
    return prefs


# User Callback implementation
# Inputs: user_input
# Outputs: node_status, ml_model_metadata
def task_callback(user_input, node_status, ml_model_metadata):
    # Callback implementation here
    print(f"Received Task: {user_input.task_id().problem_id()},{user_input.task_id().iteration_id()}")
    client = Client(host='http://localhost:11434')
    dataset_metadata = {}
    # CNNs on the Axelera board: the goal is one of the vision tasks of its YOLO models
    is_yolo = False

    try:
        extra_data_bytes = user_input.extra_data()
        extra_data_str = ''.join(chr(b) for b in extra_data_bytes)
        try:
            extra_data_dict = json.loads(extra_data_str)
        except json.JSONDecodeError:
            print("[WARN] In model_metadata node extra_data JSON is not valid.")
            extra_data_dict = {}

        accumulated_data = {}

        if "model_restrains" in extra_data_dict:
            accumulated_data["model_restrains"] = extra_data_dict["model_restrains"]

        if "model_selected" in extra_data_dict and extra_data_dict["model_selected"] != "":
            accumulated_data["model_selected"] = extra_data_dict["model_selected"]

        if "type" in extra_data_dict and extra_data_dict["type"] != "":
            accumulated_data["type"] = extra_data_dict["type"]

        family = extra_data_dict.get("model_family") or extra_data_dict.get("type") or ""
        hardware = extra_data_dict.get("hardware_required") or ""
        if hardware:
            accumulated_data["hardware_required"] = hardware
        is_yolo = family.strip().lower() == "cnns" and hw_provider_axelera.is_axelera(hardware)

        # The description picks the YOLO model too, whether the goal is chosen in the UI or inferred
        if is_yolo and not extra_data_dict.get("model_selected"):
            accumulated_data["yolo_preferences"] = get_yolo_preferences(client, user_input)

        problem = user_input.problem_short_description()
        accumulated_data["problem_short_description"] = problem

        # Persist what we’ve gathered so downstream nodes see it
        encoded_data = json.dumps(accumulated_data).encode("utf-8")
        ml_model_metadata.extra_data(encoded_data)

        ui_goal = (extra_data_dict.get("goal") or "").strip()
        if ui_goal:
            # Ensure the metadata list contains exactly this goal
            try:
                lst = ml_model_metadata.ml_model_metadata()
                lst.clear()
                lst.append(ui_goal)
            except Exception:
                # Fallback in case the API expects direct assignment
                ml_model_metadata.ml_model_metadata([ui_goal])
            print(f"[ML_MODEL_METADATA] Using UI-provided goal: {ui_goal}")
            return

        # If a specific model is chosen, bypass goal inference entirely
        if extra_data_dict.get("model_selected"):
            print(f"[ML_MODEL_METADATA] Skipping goal inference; model_selected='{extra_data_dict['model_selected']}'")
            # Optional: explicitly clear to show blank for the CNN+FPGA U-Net fast path
            try:
                ml_model_metadata.ml_model_metadata().clear()
            except Exception:
                ml_model_metadata.ml_model_metadata([])
            return

        # Collect optional dataset metadata (used only to enrich the prompt)
        if "dataset_metadata_description" in extra_data_dict:
            dataset_metadata["description"] = extra_data_dict["dataset_metadata_description"]
        if "dataset_metadata_topic" in extra_data_dict:
            dataset_metadata["topic"] = extra_data_dict["dataset_metadata_topic"]
        if "dataset_metadata_profile" in extra_data_dict:
            dataset_metadata["profile"] = extra_data_dict["dataset_metadata_profile"]
        if "dataset_metadata_keywords" in extra_data_dict:
            dataset_metadata["keywords"] = extra_data_dict["dataset_metadata_keywords"]
        if "dataset_metadata_applications" in extra_data_dict:
            dataset_metadata["applications"] = extra_data_dict["dataset_metadata_applications"]

    except Exception as e:
        print(f"No extra data was found: {e}")

    # Retrieve Possible ML Goals: the vision tasks of the YOLO models for CNNs on the Axelera
    # board, the graph goals otherwise
    try:
        if is_yolo:
            goal_descriptions = hw_provider_axelera.goal_descriptions()
            goals = list(goal_descriptions.keys())
        else:
            raw_goals = get_problems()
            inputs = [str(g) for g in raw_goals]
            goals = [goal for goal in inputs if goal not in unsupported_goals]
    except Exception as e:
        print(f"Error in getting problems from MLModel graph: {e}")
        return

    # Select MLGoal Using Ollama llama 3
    if is_yolo:
        described = "; ".join(f"{g}: {d}" for g, d in goal_descriptions.items())
        prompt = (
            f"Which of the following computer vision goals can be used to solve this problem: {described}?. "
            f"Answer with only one of the goal names ({', '.join(goals)}) and nothing more, without \"\" or ''. "
            f"If you are not sure, answer with 'None'."
        )
    else:
        prompt = (
            f"Which of the following machine learning Goals can be used to solve this problem: {goals}?. "
            f"Answer with only one of the Machine learning goals and nothing more, just the goal name without \"\" or ''. "
            f"If you are not sure, answer with 'None'."
        )
    if (user_input.inputs()):
        prompt = f"{prompt} The user inputs known are {', '.join(user_input.inputs())}."
    if (user_input.outputs()):
        prompt = f"{prompt} The user outputs known are {', '.join(user_input.outputs())}."
    if isinstance(user_input.minimum_samples(), int) and user_input.minimum_samples() > 0:
        prompt = f"{prompt} Have into account that needs to have {user_input.minimum_samples()} minimum samples."
    if isinstance(user_input.maximum_samples(), int) and user_input.maximum_samples() > 0:
        prompt = f"{prompt} Have into account that needs to have {user_input.maximum_samples()} maximum samples."
    if dataset_metadata:
        prompt = f"{prompt} The dataset that must be analyzed with the chosen Goal has the following metadata {json.dumps(dataset_metadata)}."

    problem = user_input.problem_short_description()
    if (user_input.problem_definition() != ""):
        problem = f"{problem}. {user_input.problem_definition()}."

    mlgoal = None
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        mlgoal = get_llm_response(client, "llama3", problem, prompt)
        if mlgoal:
            mlgoal = mlgoal.strip().lower()
        if mlgoal is not None and mlgoal in goals:
            break
        attempt += 1
        prompt = f"Your previous answer '{mlgoal}' was not valid. {prompt}"
        print(f"Retry {attempt}: Response '{mlgoal}' is not among available goals. Retrying...")

    if mlgoal is not None and mlgoal in goals:
        ml_model_metadata.ml_model_metadata().append(mlgoal)
        print(f"Selected ML Goal: {mlgoal}")
    else:
        print(f"Failed to determine ML goal for task {user_input.task_id()}.")
        ml_model_metadata.ml_model_metadata().clear()
        error_message = "Failed to extract metadata due to invalid or incomplete input parameters."
        error_info = {"error": error_message}
        encoded_error = json.dumps(error_info).encode("utf-8")
        ml_model_metadata.extra_data(encoded_error)


# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here
    # The GUI requests the list of goals with "modality" (the modalities are no longer used)
    if req.configuration() == "modality":
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())
        try:
            # Retrieve Possible Ml Goals from graph
            raw_goals = get_problems()
            inputs = [str(g) for g in raw_goals]
            supported_goals = [goal for goal in inputs if goal not in unsupported_goals]
            sorted_goals = ', '.join(sorted(supported_goals))  # TODO: fix overflow bug sending goals response to request

            if sorted_goals == "":
                res.success(False)
                res.err_code(1) # 0: No error || 1: Error
            else:
                res.success(True)
                res.err_code(0) # 0: No error || 1: Error
            # print(f"Available Goals: {sorted_goals}")   #debug

            res.configuration(json.dumps(dict(goals=sorted_goals)))
        except Exception as e:
            print(f"Error getting goals from request: {e}")
            res.success(False)
            res.err_code(1) # 0: No error || 1: Error

    elif 'mode_info' in req.configuration():
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())

        try:
            model = req.configuration()[len("mode_info, "):]
            details = get_model_details(model)

            if not details:
                res.success(False)
                res.err_code(1)  # 0: No error || 1: Error
            else:
                res.success(True)
                res.err_code(0)  # 0: No error || 1: Error

            # print(f"Model details for {model}: {details}")  #debug
            res.configuration(json.dumps(details))
        except Exception as e:
            print(f"Error getting model details from request: {e}")
            res.success(False)
            res.err_code(1)

    elif 'dataset_path' in req.configuration():
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())

        client = Client(host='http://localhost:11434')
        try:
            dataset_path = req.configuration()[len("dataset_path, "):]
            # print(f"Dataset path received: {dataset_path}")  #debug

            if dataset_path.endswith('.csv'):
                # Load the CSV file
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                # Load the JSON file
                with open(dataset_path) as f:
                    data = json.load(f)
                df = json_to_dataframe(data)

            sample_df, dataset_sample = get_sample(df, sample_size=10)

            # Load the semantic profiler
            semantic_profiler = SemanticProfiler(client=client, model_name="llama3")

            # Generate the basic and semantic profiles
            basic_profile, semantic_profile_part1 = dataset_profiler(df)
            semantic_profile_part2 = semantic_profiler.analyze_dataframe(sample_df)
            semantic_profile = semantic_profile_part1+'\n'+semantic_profile_part2

            # We use the basic and semantic profiles, and the dataset topic to generate the dataset description
            description_generator = DatasetDescriptionGenerator(client=client, model_name="llama3")
            _, description = description_generator.generate_description(
                                dataset_sample=dataset_sample,
                                dataset_profile=basic_profile,
                                use_profile=True,
                                semantic_profile=semantic_profile,
                                use_semantic_profile=True,
                                data_topic=None,
                                use_topic=False
                            )

            res.success(True)
            res.configuration(json.dumps(json.loads(description)))
        except Exception as e:
            res.success(False)
            res.err_code(1)  # 0: No error || 1: Error
            print(f"Error processing dataset path from request: {e}")

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
    start_time = time.time()
    loaded = False
    while time.time() - start_time < 5:
        if load_graph():
            loaded = True
            break
        time.sleep(0.1)
    if not loaded:
        print("[Error][ml_model_metadata] Graph not available")
        exit(1)
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
