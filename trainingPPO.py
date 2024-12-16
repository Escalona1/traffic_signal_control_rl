import traci
from time import sleep
import numpy as np
import random
import timeit
import os
import torch
from torch.distributions import Categorical
from ppo_agent import PPOAgent  # Assuming PPOAgent is implemented in ppo_agent.py

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NS_LEFT_GREEN = 2
PHASE_NS_LEFT_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5

class Simulation:
    def __init__(self, sumo_cmd, max_steps, green_duration, yellow_duration, num_junctions, actions_per_junction):
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_junctions = num_junctions
        self._actions_per_junction = actions_per_junction
        self._reward_episode = []
        self._queue_length_episode = []
        self.model_path = "saved_models/"  # Set to a valid directory
        os.makedirs(self.model_path, exist_ok=True)

        self.ppo_agent = PPOAgent(
            18, 
            num_junctions=num_junctions,
            actions_per_junction=actions_per_junction
        )

        # Load the model if it exists
        model_file = os.path.join(self.model_path, "ppo79844.pth")
        if os.path.exists(model_file):
            try:
                print(f"Loading model from {model_file}")
                self.ppo_agent.policy_network.load_state_dict(torch.load(model_file))
                self.ppo_agent.policy_network.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Proceeding with an untrained model.")
        else:
            print("No pre-trained model found, initializing new agent.")

    def save_model(self, name):
        """Save the current model to the specified path."""
        model_path = os.path.join(self.model_path, name)
        print(f"Saving model to {model_path}")
        torch.save(self.ppo_agent.policy_network.state_dict(), model_path)

    def run(self, episode, train):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()
        traci.start(self._sumo_cmd)
        print("Simulating...")

        rollout_length = 300
        ep = 0
        while self._step < self._max_steps:
            states, actions, rewards, log_probs, advantages, returns = self.collect_rollout(rollout_length)

            if train:
                policy_loss, value_loss, entropy = self.ppo_agent.update(states, actions, log_probs, advantages, returns)
                print(f"Episode {ep}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
                print(self._step)
                if ep % 2 == 0:
                    self.save_model(f"ppo{self._step}.pth")
                    print('Model saved.')
            else:
                print(f"Episode {ep} completed without training.")

            ep += 1

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time

    def collect_rollout(self, rollout_length):
        states, actions, rewards, log_probs, values = [], [], [], [], []
        state = self._get_state()  # Initial state

        old_action = [0, 0, 0]

        for _ in range(rollout_length):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs, value = self.ppo_agent.policy(state_tensor)

            actions_per_junction = []
            log_probs_per_junction = []

            # Sample actions for each junction
            for junction_probs in action_probs[0]:
                dist = Categorical(junction_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                #print(log_prob)
                actions_per_junction.append(action.item())
                log_probs_per_junction.append(log_prob)

            # Execute actions in the simulation (one per junction)
            junctions = ['J1', 'J2', 'J3']
            #print(actions_per_junction)
            for i, junction in enumerate(junctions):
                if self._step != 0 and old_action[i] != actions_per_junction[i]:
                    self._set_yellow_phase(old_action[i], junction)
                    self._simulate(self._yellow_duration)

                self._set_green_phase(actions_per_junction[i], junction)

            self._simulate(self._green_duration)

            old_action[i] = actions_per_junction[i]

            self._simulate(self._green_duration)  # Simulate green phase

            reward = -self._collect_waiting_times()  # Use negative waiting times as reward
            self._reward_episode.append(reward)

            next_state = self._get_state()

            # Store trajectory data
            states.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            actions.append(torch.tensor(actions_per_junction, dtype=torch.int64).unsqueeze(0))
            log_probs.append(torch.tensor(log_probs_per_junction, dtype=torch.float32).unsqueeze(0))
            rewards.append(reward)
            values.append(value)

            # Update state
            state = next_state
        # Compute advantages and returns
        next_values = values[1:] + [torch.tensor(0.0)]
        advantages, returns = self.ppo_agent.compute_advantages(rewards, values, next_values, [0] * len(rewards))

        return (
            torch.cat(states),
            torch.cat(actions),
            rewards,
            torch.cat(log_probs),
            advantages,
            returns,
        )

    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step
            print('no more steps')

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1

            junctions = ['J1', 'J2', 'J3']

            queue_length = 0
            for junction in junctions:
                queue_length += self._get_queue_length(junction)

            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        car_list = traci.vehicle.getIDList()
        wait_time = 0

        for car_id in car_list:
            wait_time += traci.vehicle.getAccumulatedWaitingTime(car_id)

        return wait_time

    def _set_yellow_phase(self, old_action, tl_id):
        """
        Dynamically activate the correct yellow phase for the given traffic light.
        """
        phase_list = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
        yellow_phase_code = old_action * 2 + 1
        if yellow_phase_code < len(phase_list[0].phases):  # Check if the yellow phase exists
            traci.trafficlight.setPhase(tl_id, yellow_phase_code)
        else:
            print(f"Yellow phase for action {old_action} does not exist at {tl_id}. Setting all red.")
            traci.trafficlight.setPhase(tl_id, len(phase_list[0].phases) - 1)  # Default to all-red phase

    def _set_green_phase(self, action_number, tl_id):
        """
        Dynamically activate the correct green phase for the given traffic light.
        """
        # Map action numbers to green phases specific to this intersection
        phase_map = {
            "J1": {0: PHASE_NS_GREEN, 1: PHASE_NS_LEFT_GREEN, 2: PHASE_EW_GREEN},
            "J2": {0: PHASE_NS_GREEN, 1: PHASE_NS_LEFT_GREEN, 2: PHASE_EW_GREEN},
            "J3": {0: PHASE_NS_GREEN, 1: PHASE_NS_LEFT_GREEN, 2: PHASE_EW_GREEN},
        }

        if action_number in phase_map[tl_id]:
            traci.trafficlight.setPhase(tl_id, phase_map[tl_id][action_number])
        else:
            print(f"Invalid action {action_number} for {tl_id}. Setting default phase.")
            traci.trafficlight.setPhase(tl_id, PHASE_NS_GREEN)  # Default to NS green

    def _get_queue_length(self, intersection_id):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        edges = traci.junction.getIncomingEdges(intersection_id)
        halt = 0
        for edge in edges:
            halt += traci.edge.getLastStepHaltingNumber(edge)
        return halt

    def _get_state(self):
        state = []

        lane_groups = ['-E1_0', '-E1_1', '-E2_0', '-E2_1', 'E0_0', 'E0_1', 
                           'E1_0', 'E1_1', 'E2_0', 'E2_1', 'E3_0', 'E3_1',
                           'E4_0', 'E4_1', 'E6_0', 'E6_1', 'E8_0', 'E8_1']

        for lane in lane_groups:
            autos_detenidos = traci.lane.getLastStepHaltingNumber(lane)
            state.append(autos_detenidos)

        return state

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode