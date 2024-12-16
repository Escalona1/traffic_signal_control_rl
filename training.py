import traci
import numpy as np
import random
import timeit
import os
from training_red1 import DQN
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NS_LEFT_GREEN = 2
PHASE_NS_LEFT_YELLOW= 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5

class Simulation:
    def __init__(self, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        #self._Model = Model
        #self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []
        self.dqn =DQN()


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        #self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_action = [0,0,0] # dummy init
        iterator = 0
        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            """current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait"""
            #action = random.randint(0,2)
            action = self.dqn.select_action(current_state)
            # if the chosen phase is different from the last phase, activate the yellow phase
            junctions = ['J1', 'J2', 'J3']

            i = 0
            for junction in junctions:

                if self._step != 0 and old_action[i] != action[i]:
                    self._set_yellow_phase(old_action[i], junction)
                    self._simulate(self._yellow_duration)

                # execute the phase selected before
                self._set_green_phase(action[i], junction)
                self._simulate(self._green_duration)

                # saving variables for later & accumulate reward
                old_action[i] = action[i]
                
                i+=1
            #train
            new_state = self._get_state()
            reward = -self._collect_waiting_times()
            
            iterator += 1
            
            self.dqn.train(iterator,current_state,new_state,reward,action[0],action[1],action[2])

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
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
            wait_time -= traci.vehicle.getAccumulatedWaitingTime(car_id)
        
        return wait_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


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
            #autos_fila = traci.lane.getLastStepVehicleNumber(lane)
            autos_detenidos = traci.lane.getLastStepHaltingNumber(lane)
            state.append(autos_detenidos)
        
        return state


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



