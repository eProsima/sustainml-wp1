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

from sustainml_py.nodes.TaskEncoderNode import TaskEncoderNode

# Manage signaling
import signal
import threading
import time

# Whether to go on spinning or interrupt
running = False

# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    TaskEncoderNode.terminate()
    global running
    running = False

# User Callback implementation
# Inputs: user_input
# Outputs: node_status, encoded_task
def task_callback(user_input, node_status, encoded_task):
    # Callback implementation here
    print (user_input.problem_description())
    # include short delay for demos
    time.sleep(2)
    # populate output
    encoded_task.keywords().append("User")
    encoded_task.keywords().append("input")
    encoded_task.keywords().append("detected")
    encoded_task.keywords().append("keywords")
    encoded_task.keywords().append("example")

# Main workflow routine
def run():
    node = TaskEncoderNode(callback=task_callback)
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
