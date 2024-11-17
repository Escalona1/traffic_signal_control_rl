import traci
import numpy as np
import random
import timeit
import os

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
        old_total_wait = 0
        old_action = [0,0,0] # dummy init

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            #action = self._choose_action(current_state)
            action = random.randint(0,2)
            # if the chosen phase is different from the last phase, activate the yellow phase

            junctions = ['J1', 'J2', 'J3']

            i = 0
            for junction in junctions:

                if self._step != 0 and old_action[i] != action:
                    self._set_yellow_phase(old_action[i], junction)
                    self._simulate(self._yellow_duration)

                # execute the phase selected before
                self._set_green_phase(action, junction)
                self._simulate(self._green_duration)

                # saving variables for later & accumulate reward
                old_action[i] = action
                old_total_wait = current_total_wait
                i+=1

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
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = []
        junctions = traci.junction.getIDList()

        for junction in junctions:
            incoming_roads.append(traci.junction.getIncomingEdges(junction))

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


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
        halt =0
        for edge in edges:
            halt+= traci.edge.getLastStepHaltingNumber(edge)
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        """
        return halt


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(200000)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            lane_cell = lane_pos//7
            lane_cell = int(lane_cell)
            
            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes

            lane_groups = ['-E1_0', '-E1_1', '-E2_0', '-E2_1', 'E0_0', 'E0_1', 
                           'E1_0', 'E1_1', 'E2_0', 'E2_1', 'E3_0', 'E3_1',
                           'E4_0', 'E4_1', 'E6_0', 'E6_1', 'E8_0', 'E8_1']

            for group_index, lanes in enumerate(lane_groups):
                if lane_id in lanes:
                    lane_group = group_index
                    break
                else:
                    lane_group = -1


            if lane_group >= 1 and lane_group <= 18:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



