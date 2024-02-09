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
import signal
import threading
import time

# Whether to go on spinning or interrupt
running = False

# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    MLModelNode.terminate()
    global running
    running = False

# User Callback implementation
# Inputs: encoded_task
# Outputs: node_status, ml_model
def task_callback(encoded_task, node_status, ml_model):
    # Callback implementation here
    for keyword in encoded_task.keywords():
        print("Received keyword " + keyword)
    # include short delay for demos
    time.sleep(2)
    # populate output
    ml_model.model("MODEL in ONXX format should be here")

# Main workflow routine
def run():
    node = MLModelNode(callback=task_callback)
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
