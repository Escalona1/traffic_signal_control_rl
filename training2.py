import traci
import numpy as np
import torch
import timeit
from training_red2 import DQNsharedbasedNetwork

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NS_LEFT_GREEN = 2
PHASE_NS_LEFT_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5

import traci
import numpy as np
import timeit
import os
from training_red2 import DQNsharedbasedNetwork

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NS_LEFT_GREEN = 2
PHASE_NS_LEFT_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5

class Simulation:
    def __init__(self, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []
        self.model_path = "saved_models/"  # Set to a valid directory
        os.makedirs(self.model_path, exist_ok=True)

        # Initialize the DQN agent
        self.dqn = DQNsharedbasedNetwork()

        # Load the model if it exists
        model_file = os.path.join(self.model_path, "dqn_red2_model_step_189044.pth")
        if os.path.exists(model_file):
            try:
                print(f"Loading model from {model_file}")
                self.dqn.policy_network.load_state_dict(torch.load(model_file))
                self.dqn.policy_network.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Proceeding with an untrained model.")
        else:
            print("No pre-trained model found, initializing new agent.")

    def save_model(self, name):
        """Save the current model to the specified path."""
        model_path = os.path.join(self.model_path, name)
        print(f"Saving model to {model_path}")
        torch.save(self.dqn.policy_network.state_dict(), model_path)

    def run(self, episode, train=True):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        old_action = [0, 0, 0]  # Dummy initialization
        iterator = 0
        ep = 0

        while self._step < self._max_steps:
            current_state = self._get_state()
            action = self.dqn.select_action(current_state)
            junctions = ['J1', 'J2', 'J3']

            for i, junction in enumerate(junctions):
                if self._step != 0 and old_action[i] != action[i]:
                    self._set_yellow_phase(old_action[i], junction)
                    self._simulate(self._yellow_duration)

                self._set_green_phase(action[i], junction)

            self._simulate(self._green_duration)

            old_action[i] = action[i]
            new_state = self._get_state()
            reward = self._collect_waiting_times()

            iterator += 1
            if train:
                self.dqn.train(iterator, current_state, new_state, reward, action[0], action[1], action[2])
            self._reward_episode.append(reward)

            # Save model periodically
            if train and ep % 200 == 0:
                self.save_model(f"dqn_red2_model_step_{self._step}.pth")
                print("Model saved.")
            ep += 1

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time


    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1

            queue_length = sum(
                self._get_queue_length(junction) for junction in ['J1', 'J2', 'J3']
            )
            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):

        car_list = traci.vehicle.getIDList()
        wait_time = 0
        
        for car_id in car_list:
            wait_time -= traci.vehicle.getAccumulatedWaitingTime(car_id)
        
        return wait_time

    def _set_yellow_phase(self, old_action, tl_id):
        phase_list = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
        yellow_phase_code = old_action * 2 + 1
        if yellow_phase_code < len(phase_list[0].phases):
            traci.trafficlight.setPhase(tl_id, yellow_phase_code)
        else:
            traci.trafficlight.setPhase(tl_id, len(phase_list[0].phases) - 1)

    def _set_green_phase(self, action_number, tl_id):
        phase_map = {
            "J1": {0: PHASE_NS_GREEN, 1: PHASE_NS_LEFT_GREEN, 2: PHASE_EW_GREEN},
            "J2": {0: PHASE_NS_GREEN, 1: PHASE_NS_LEFT_GREEN, 2: PHASE_EW_GREEN},
            "J3": {0: PHASE_NS_GREEN, 1: PHASE_NS_LEFT_GREEN, 2: PHASE_EW_GREEN},
        }
        traci.trafficlight.setPhase(tl_id, phase_map.get(tl_id, {}).get(action_number, PHASE_NS_GREEN))

    def _get_queue_length(self, intersection_id):
        return sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.junction.getIncomingEdges(intersection_id))

    def _get_state(self):
        state = []
        lane_groups = ['-E1_0', '-E1_1', '-E2_0', '-E2_1', 'E0_0', 'E0_1',
                       'E1_0', 'E1_1', 'E2_0', 'E2_1', 'E3_0', 'E3_1',
                       'E4_0', 'E4_1', 'E6_0', 'E6_1', 'E8_0', 'E8_1']

        for lane in lane_groups:
            autos_fila = traci.lane.getLastStepVehicleNumber(lane)
            autos_detenidos = traci.lane.getLastStepHaltingNumber(lane)
            state.extend([autos_fila, autos_detenidos])

        return np.array(state, dtype=np.float32)

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode
