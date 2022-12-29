import ast
import itertools
import time
from copy import deepcopy
from typing import Tuple

from sqlalchemy.dialects.mysql import ENUM

from inputs import small_inputs
from additional_inputs import additional_inputs
import random

# import networkx as nx
import numpy as np

from utils import orientations, powerset, vector_add

ids = ["316375872", "206014482"]

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1
END_OF_GAME_STATE = "EndOfGame"


class OptimalTaxiAgent:
    def __init__(self, initial: dict):
        self.initial = initial
        # we shouldn't add anything to the initial state! because we get the state from user in "act" func
        self.max_turns_to_go = initial["turns to go"]
        self.max_fuel_per_taxi = self.get_max_fuel_per_taxi(initial)
        self.max_capacity_per_taxi = self.get_max_capacity_per_taxi(initial)
        self.n_passengers = len(initial["passengers"].keys())
        self.map_num_rows = len(initial["map"])
        self.map_num_cols = len(initial["map"][0])

        self.all_possible_states = []
        self.all_possible_actions = []  # list [a1, a2, ...]
        self.all_possible_actions_for_state = {}  # dict {state: a1, a2, ...}

        # for Value Iteration
        self.Next_States_Probs = {}
        self.Rewards = {}
        self.Actions_Values = {}
        self.Values = {}
        self.Best_Actions = {}

        self.set_all_possible_states()
        self.set_all_possible_actions_and_next_states()
        self.set_rewards_for_actions()
        self.value_iteration_with_dicts()

    def act(self, state: dict):
        """
        Get the best action by best_actions dict.
        best actions dict was precalculated by value iteration in the init of agent
        """
        state = deepcopy(state)
        t = state["turns to go"]
        state.pop("turns to go")  # remove t from state
        state_str = dict_to_str(state)
        best_action = self.Best_Actions[t][state_str]
        if t == self.max_turns_to_go:    # for debug
            print(f"{t=}", self.Values[t][state_str])
            # for t_ in reversed(range(1, 101)):        # for debug
            #     print(f"{t_=}", self.Values[t_][state_str])     # for debug
            # print()
        # print(f"{t=}", self.Best_Actions[t][state_str], self.Values[t][state_str], state['passengers']['Dana']['destination'])  # for debug
        return best_action

    def set_all_possible_states(self):
        """
        Get all possible permutations of states
        Retruns list of states as json strings.
        [s1, s2, s3, ...]
        states parameters are: optimal, game_map, taxis_option, passengers_option
            (and not including turns to go)
        """
        optimal = self.initial["optimal"]
        game_map = self.initial["map"]
        taxis_init = self.initial["taxis"]
        passengers_init = self.initial["passengers"]
        all_taxis_options = self.get_all_possible_taxis_dicts(taxis_init, game_map)
        all_passengers_options = self.get_all_possible_passengers_dicts(
            passengers_init, taxis_init
        )

        all_possible_states = []
        for taxis_option in all_taxis_options:
            for passengers_option in all_passengers_options:
                # create state dict
                state = {
                    "optimal": optimal,
                    "map": game_map,
                    "taxis": taxis_option,
                    "passengers": passengers_option,
                }
                # checl legal capacity and passenger location
                if self.is_legal_capacity_and_passengers_locations(state):
                    all_possible_states.append(dict_to_str(state))
        all_possible_states.append(END_OF_GAME_STATE)
        self.all_possible_states = all_possible_states

    def is_legal_capacity_and_passengers_locations(self, state):
        """
        example: num_passengers in taxi   = 3
                 max_capacity of taxi     = 4
                 current_capacity of taxi = 1
                 current_capacity + num_passengers =? max_capacity
                 1 + 3 =? 4 ==> is legal!
        """
        legal = True
        num_passengers_per_taxi = {taxi_name: 0 for taxi_name in state["taxis"].keys()}
        for pass_dict in state["passengers"].values():
            pass_location = pass_dict["location"]
            if pass_location in state["taxis"]:
                num_passengers_per_taxi[pass_location] += 1
        for taxi_name, taxi_details in state["taxis"].items():
            if (
                taxi_details["capacity"] + num_passengers_per_taxi[taxi_name]
                != self.max_capacity_per_taxi[taxi_name]
            ):
                legal = False
        return legal

    def get_all_possible_taxis_dicts(self, taxis_init: dict, game_map: list) -> list:
        """
        get all possible taxis dicts
        :param dict taxis_init: taxis dict in initial state of game.
        :param list game_map:
        """
        all_options_by_taxi = {}
        # taxi possible locations is the same for all taxis
        possible_locations = self.get_all_legal_taxis_locations(game_map)
        for taxi_name, taxi_details in taxis_init.items():
            all_options_by_taxi[taxi_name] = []
            possible_fuel_vals = range(taxi_details["fuel"] + 1)
            possible_cacpacity_vals = range(taxi_details["capacity"] + 1)
            for location in possible_locations:
                for fuel_value in possible_fuel_vals:
                    for capacity_value in possible_cacpacity_vals:
                        taxi_state = {
                            taxi_name: {
                                "location": location,
                                "fuel": fuel_value,
                                "capacity": capacity_value,
                            }
                        }
                        all_options_by_taxi[taxi_name].append(taxi_state)

        # Get all permutations of taxis dicts (permutation contains one dict of each taxi)
        taxis_options_lists = list(all_options_by_taxi.values())
        if len(taxis_options_lists) == 1:
            all_taxis_permutations = [[i] for i in taxis_options_lists[0]]
        else:
            all_taxis_permutations = list(itertools.product(*taxis_options_lists))

        # create taxis dict from each permutation of dicts
        all_taxis_options = []
        for permutation in all_taxis_permutations:
            taxis_dict = {}
            for i, taxi_name in enumerate(taxis_init.keys()):
                taxis_dict[taxi_name] = permutation[i][taxi_name]
            all_taxis_options.append(taxis_dict)
        return all_taxis_options

    def get_all_legal_taxis_locations(self, game_map):
        """
        taxi location is any location on game_map that is not "I"
        """
        taxis_locations = [
            (i, j) for i in range(self.map_num_rows) for j in range(self.map_num_cols)
        ]
        for x, y in taxis_locations:
            # remove I tiles
            if game_map[x][y] == "I":
                taxis_locations.remove((x, y))
        return taxis_locations

    def get_all_possible_passengers_dicts(self, passengers_init, taxis_init):
        """
        get all possible passengers dicts
        :param dict passengers_init: passengers dict in initial state of game.
        :param dict taxis_init: taxis dict in initial state of game.
        """
        all_options_by_passenger = {}
        # passenger possible location depends on passenger params
        all_legal_locations_by_passenger = self.get_all_legal_locations_by_passenger(
            passengers_init, taxis_init
        )
        for passenger_name, passenger_details in passengers_init.items():
            all_options_by_passenger[passenger_name] = []
            possible_locations = all_legal_locations_by_passenger[passenger_name]
            possible_destinations = passenger_details["possible_goals"]
            # possible_goals and prob_change_goal do not change between states
            possible_goals = passenger_details["possible_goals"]
            prob_change_goal = passenger_details["prob_change_goal"]
            for location in possible_locations:
                for destination in possible_destinations:
                    passenger_state = {
                        passenger_name: {
                            "location": location,
                            "destination": destination,
                            "possible_goals": possible_goals,
                            "prob_change_goal": prob_change_goal,
                        }
                    }
                    all_options_by_passenger[passenger_name].append(passenger_state)

        # Get all permutations of passengers dicts (permutation contains one dict of each passenger)
        passengers_options_lists = list(all_options_by_passenger.values())
        if len(passengers_options_lists) == 1:
            all_passengers_permutations = [[i] for i in passengers_options_lists[0]]
        else:
            all_passengers_permutations = list(
                itertools.product(*passengers_options_lists)
            )

        # create passengers dict from each permutation of dicts
        all_passengers_options = []
        for permutation in all_passengers_permutations:
            passengers_dict = {}
            for i, pass_name in enumerate(passengers_init.keys()):
                passengers_dict[pass_name] = permutation[i][pass_name]
            all_passengers_options.append(passengers_dict)
        return all_passengers_options

    def get_all_legal_locations_by_passenger(self, passengers_init, taxis_init):
        """
        passenger location is one of the following:
            1. the initialized location of this passenger
            2. one of the taxis names (when taxi picked him up)
            3. one of the possible goals of this passenger (when taxi dropped him offß)
        """
        taxis_names = list(taxis_init.keys())
        all_legal_locations_by_passenger = {}
        for curr_passenger, passengers_dict in passengers_init.items():
            init_location = [tuple(passengers_dict["location"])]
            possible_goals = list(passengers_dict["possible_goals"])
            all_legal_locations_by_passenger[curr_passenger] = set(
                init_location + taxis_names + possible_goals
            )
        return all_legal_locations_by_passenger

    def set_all_possible_actions_and_next_states(self):
        """
        Update:
        @ list: self.all_possible_actions_for_state
        @ list: self.all_possible_actions
        @ dict: self.Next_States_Probs {(state0, action1):
                                            [(state1, prob1), (state2, prob2)]}
                For each possible state, checks the possible next states
                with each action and the probability to get to this next step.
        """
        for state in self.all_possible_states:
            legal_actions = self.actions(state)
            self.all_possible_actions_for_state[state] = legal_actions
            if len(legal_actions) == 0:
                continue
            if state == END_OF_GAME_STATE:
                continue
            for action in legal_actions:
                if action not in self.all_possible_actions:
                    self.all_possible_actions.append(action)
                # find next_states and probs
                if action == "terminate":
                    self.Next_States_Probs[(state, action)] = ()
                    continue
                next_states_with_probs = self.result_with_probs(state, action)
                self.Next_States_Probs[(state, action)] = []
                for next_state, prob in next_states_with_probs.items():
                    self.Next_States_Probs[(state, action)].append((next_state, prob))

    def result_with_probs(self, state, action):
        """
        input: state, action
        output: next possible states and the probability for each one
        method:
            1.  Use the deterministic result function to get the next "regular" state.
            2.  Get all possible new states with possible changes of destinations
                2a. Get new_state prob and possible destinations
                    where the subset passengers may change their destination
                2b. Get all destinations permutations
                    (passengers have few possible goals to change to)
                2c. For each permutation get new state with updated destinations

        """
        possible_next_state_probs = {}
        # 1. Use the deterministic result function to get the next "regular" state.
        result_state_str = self.result(state, action)
        # 2. Get all possible new states with possible changes of destinations
        if action == "reset":
            reset_prob = 1  # we saw in check.py that they are not doing rechoice
            possible_next_state_probs[result_state_str] = reset_prob
            return possible_next_state_probs

        result_state = str_to_dict(result_state_str)
        all_passengers_subsets = self.get_all_passengers_subsets(result_state)

        for pass_subset_names in all_passengers_subsets:
            # 2a. Get new_state prob and possible destinations lists
            #     where the subset passengers may change their destination
            new_state_prob, dest_lists = self.get_prob_and_dest_lists(
                result_state, pass_subset_names
            )

            # 2b. Get all destinations permutations
            all_dest_permu = self.get_all_dest_permutations(dest_lists)

            # 2c. For each permutation get new state with updated destinations
            #     and update possible_next_state_probs dict
            for dest_permu in all_dest_permu:
                new_state = self.get_new_state(result_state, dest_permu)
                possible_next_state_probs = self.add_state_and_prob(
                    new_state, new_state_prob, possible_next_state_probs
                )

        return possible_next_state_probs

    def get_prob_and_dest_lists(self, result_state, pass_subset_names):
        new_state_prob = 1  # init prob
        dest_lists = []
        for pass_name, pass_details in result_state["passengers"].items():
            if pass_name in pass_subset_names:  # passenger rechoice destination
                possible_goals = list(pass_details["possible_goals"])
                n_goals = float(len(possible_goals))
                new_state_prob *= (pass_details["prob_change_goal"]) / n_goals
            else:
                possible_goals = list([pass_details["destination"]])
                new_state_prob *= 1.0 - pass_details["prob_change_goal"]
            dest_lists.append(possible_goals)
        return new_state_prob, dest_lists

    def get_all_passengers_subsets(self, result_state):
        if self.n_passengers == 1:
            all_passengers_subsets = list(result_state["passengers"].keys())
        else:
            all_passengers_subsets = powerset(result_state["passengers"].keys())
        all_passengers_subsets.append(
            [()]
        )  # no passenger is rechoicing his destination
        return all_passengers_subsets

    def get_all_dest_permutations(self, possible_dest_lists):
        if len(possible_dest_lists) <= 1:
            all_dest_permu = possible_dest_lists[0]
        else:
            all_dest_permu = list(itertools.product(*possible_dest_lists))
        return all_dest_permu

    def get_new_state(self, result_state, dest_permu):
        """
        @ dest_permu: destinations permutation
            example [(0,0), (2,1), (3,3)] permutaion of destinations
                for [pass1, pass2, pass3]
        """
        new_state = deepcopy(result_state)
        if self.n_passengers == 1:
            for pass_name in new_state["passengers"].keys():
                new_state["passengers"][pass_name]["destination"] = dest_permu
        else:
            for i, pass_name in enumerate(new_state["passengers"].keys()):
                new_state["passengers"][pass_name]["destination"] = dest_permu[i]
        return new_state

    def add_state_and_prob(self, new_state, new_state_prob, possible_next_state_probs):
        new_state_str = dict_to_str(new_state)
        if new_state_str in possible_next_state_probs.keys():  # sum probs
            existing_prob = possible_next_state_probs[new_state_str]
            additional_prob = new_state_prob
            possible_next_state_probs[new_state_str] = existing_prob + additional_prob
        else:
            possible_next_state_probs[new_state_str] = new_state_prob
        return possible_next_state_probs

    def value_iteration_with_dicts(self):
        """
        Get the best action for each step t
        from t = 0 to t = max_t.
        Update self.Best_Actions dict.
        Method: running value iteration algorithm.
        Using:
        @ Next_States_Probs: dict {(state0, action1):
                                    [(state1, prob1), (state2, prob2)]}
        @ Rewards: dict {(action): reward}  # depends only on action!!
        @ Actions_Values    {t:
                                {state0:
                                    {(action1): value,
                                     (action2): value}
                                {state1}:
                                    {(action1): value,
                                     (action2): value}
                            }
        @ Best_Actions:     {t:
                                {state0: best_action,
                                 state1: best_action}
                            }
        @ Values            {t:
                                {state0: best_value,
                                 state1: best_value}
                            }
        """
        for t in range(self.max_turns_to_go + 1):
            self.Values[t] = {}
            self.Actions_Values[t] = {}
            self.Best_Actions[t] = {}
            if t == 0:
                # End of game - no action is possible --> value is 0 for all states.
                for state in self.all_possible_states:
                    self.Values[t][state] = 0
            else:
                for state in self.all_possible_states:
                    if state == END_OF_GAME_STATE:
                        self.Values[t][state] = 0
                        continue
                    self.Actions_Values[t][state] = {}
                    # Update - Actions_Values dict
                    for action in self.all_possible_actions_for_state[state]:
                        # V(t, s) = max_over_a{
                        #              sum_over_next_s[prob(next_s, a)* V(t-1, next_s)] + Reward(a)}
                        value_of_action = self.Rewards[action]  # init
                        for next_state, prob in self.Next_States_Probs[(state, action)]:
                            value_of_action += prob * self.Values[t - 1][next_state]    #
                        self.Actions_Values[t][state][action] = value_of_action
                    # Find - best action and best value
                    possible_actions = list(self.Actions_Values[t][state].keys())
                    possible_values = list(self.Actions_Values[t][state].values())
                    best_value = max(possible_values)
                    best_action = possible_actions[np.argmax(possible_values)]
                    # Update  - Values and Best_Actions dicts
                    self.Values[t][state] = best_value
                    self.Best_Actions[t][state] = best_action

    def set_rewards_for_actions(self):
        """
        For each action set the reward.
        """
        for action in self.all_possible_actions:
            if action == "reset":
                reward = -RESET_PENALTY
            elif action == "terminate":
                reward = 0
            else:
                reward = 0
                for atomic_action in action:
                    if atomic_action[0] == "drop off":
                        reward += DROP_IN_DESTINATION_REWARD
                    elif atomic_action[0] == "refuel":
                        reward -= REFUEL_PENALTY
            self.Rewards[action] = reward

    def get_max_fuel_per_taxi(self, initial):
        max_fuel_per_taxi = {}
        for taxi_name, taxi_dict in initial["taxis"].items():
            max_fuel_per_taxi[taxi_name] = taxi_dict["fuel"]
        return max_fuel_per_taxi

    def get_max_capacity_per_taxi(self, initial):
        max_capacity_per_taxi = {}
        for taxi_name, taxi_dict in initial["taxis"].items():
            max_capacity_per_taxi[taxi_name] = taxi_dict["capacity"]
        return max_capacity_per_taxi

    def get_gas_station_list(self, map, h, w):
        g_list = []
        for i in range(h):
            for j in range(w):
                if map[i][j] == "G":
                    g_list.apend((i, j))
        return g_list

    def generate_locations(self, state: dict) -> dict:
        # get new locations by:
        # current location + one step in legal orientation (EAST, NORTH, WEST, SOUTH)
        possible_locations_by_taxi = dict()
        for taxi_name, taxi_dict in state["taxis"].items():
            curr_location = taxi_dict["location"]
            possible_locations = [
                vector_add(curr_location, orient) for orient in orientations
            ]
            possible_locations_by_taxi[taxi_name] = possible_locations

        return possible_locations_by_taxi

    def get_legal_moves_on_map(self, state: dict) -> dict:
        legal_locations_by_taxi = {}
        possible_locations_by_taxi = self.generate_locations(state)
        for taxi_name, taxi_dict in state["taxis"].items():
            # 1. check fuel > 0
            legal_locations = []
            if taxi_dict["fuel"] > 0:
                possible_locations = possible_locations_by_taxi[taxi_name]
                for new_location in possible_locations:
                    x, y = new_location
                    # 2. check that the taxi doesn't get out of the map
                    # 3. check that the taxi is on a passable tile
                    if (0 <= x < self.map_num_rows) and (0 <= y < self.map_num_cols):
                        if state["map"][x][y] != "I":
                            legal_locations.append(new_location)
            legal_locations_by_taxi[taxi_name] = legal_locations
        return legal_locations_by_taxi

    def get_legal_refuel(self, state: dict) -> dict:
        # Refueling can be performed only at gas stations
        legal_refuels_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            map_matrix = state["map"]
            x, y = taxi_dict["location"]  # current location of taxi
            # check that the location on map is "G"
            legal_refuel = map_matrix[x][y] == "G"  # bool
            legal_refuels_by_taxi[taxi_name] = legal_refuel
        return legal_refuels_by_taxi

    def get_legal_pick_up(self, state: dict) -> dict:
        # Pick up passengers if they are on the same tile as the taxi.
        legal_pickups_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            capacity = taxi_dict["capacity"]
            legal_pickups = []
            # The number of passengers in the taxi has to be < taxi’s capacity.
            if capacity > 0:
                for passenger_name, passenger_dict in state["passengers"].items():
                    # check that location of taxi is the same as location of the passenger
                    # and check that the location of the passenger is not his destination
                    if (taxi_dict["location"] == passenger_dict["location"]) & (
                        passenger_dict["location"] != passenger_dict["destination"]
                    ):
                        legal_pickups.append(passenger_name)
            legal_pickups_by_taxi[taxi_name] = legal_pickups
        return legal_pickups_by_taxi

    def get_legal_drop_off(self, state: dict) -> dict:
        # The passenger can only be dropped off on his destination tile
        # and will refuse to leave the vehicle otherwise.
        legal_drop_offs_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            legal_drop_offs = []
            # go over the passengers that's on the curr taxi
            for passenger_name, passenger_dict in state["passengers"].items():
                if passenger_dict["location"] == taxi_name:
                    # check that location of taxi is the same as destination of the passenger
                    if taxi_dict["location"] == passenger_dict["destination"]:
                        legal_drop_offs.append(passenger_name)
            legal_drop_offs_by_taxi[taxi_name] = legal_drop_offs
        return legal_drop_offs_by_taxi

    def actions(self, state: str) -> Tuple[Tuple[Tuple]]:
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        # -----------------------------------------------------------------
        # Atomic Actions: ["move", "pick_up", "drop_off", "refuel", "wait"]
        # explicit syntax:
        # (“move”, “taxi_name”, (x, y))
        # (“pick up”, “taxi_name”, “passenger_name”
        # (“drop off”, “taxi_name”, “passenger_name”)
        # ("refuel", "taxi_name")
        # ("wait", "taxi_name")

        # Full Action - a tuple with action for each taxi
        # Example: ((“move”, “taxi 1”, (1, 2)),
        #           (“wait”, “taxi 2”),
        #           (“pick up”, “very_fancy_taxi”, “Yossi”))
        # -----------------------------------------------------------------

        if state == END_OF_GAME_STATE:
            legal_actions = tuple()
            return legal_actions
        state = str_to_dict(state)

        # For each taxi get Possible Atomic Actions

        legal_locations_by_taxi = self.get_legal_moves_on_map(
            state
        )  # DICT[taxi_name: list of (x,y) locations]
        legal_pickups_by_taxi = self.get_legal_pick_up(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_drop_offs_by_taxi = self.get_legal_drop_off(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_refuels_by_taxi = self.get_legal_refuel(
            state
        )  # DICT[taxi_name: True / False]

        # -----------------------------------------------------------------
        # Get Atomic Actions with right syntax
        atomic_actions_lists = []
        for taxi_name in state["taxis"].keys():
            atomic_actions = [("wait", taxi_name)]
            for location in legal_locations_by_taxi[taxi_name]:
                atomic_actions.append(("move", taxi_name, location))
            for passenger_name in legal_pickups_by_taxi[taxi_name]:
                atomic_actions.append(("pick up", taxi_name, passenger_name))
            for passenger_name in legal_drop_offs_by_taxi[taxi_name]:
                atomic_actions.append(("drop off", taxi_name, passenger_name))
            if legal_refuels_by_taxi[taxi_name]:
                atomic_actions.append(("refuel", taxi_name))
            atomic_actions_lists.append(atomic_actions)

        # -----------------------------------------------------------------
        # Get Actions - all permutations of atomic actions
        actions = list(itertools.product(*atomic_actions_lists))
        # all_wait_action = tuple(
        #     [("wait", taxi_name) for taxi_name in state["taxis"].keys()]
        # )
        # # assert all_wait_action in actions
        # # actions.remove(all_wait_action)

        # -----------------------------------------------------------------
        # For each action - Check That Taxis Don't Clash with each other
        #   == not going to the same location (therefore cannot pickup the same passenger)
        if len(state["taxis"]) > 1:
            legal_actions = []
            for action in actions:
                taxis_next_locations = []
                for (
                    atomic_action
                ) in action:  # TODO: NOTE changed from atomic_actions_lists to action
                    action_type = atomic_action[0]
                    taxi_name = atomic_action[1]
                    taxi_curr_location = state["taxis"][taxi_name]["location"]
                    if action_type == "move":
                        taxi_next_location = atomic_action[2]
                    else:
                        taxi_next_location = taxi_curr_location
                    taxis_next_locations.append(taxi_next_location)
                # check if there are 2 taxis in the same location
                legal_action = len(set(taxis_next_locations)) == len(state["taxis"])
                if legal_action:
                    legal_actions.append(action)
        else:  # n_taxis == 1 --> no clashing between taxis
            legal_actions = actions

        # -----------------------------------------------------------------
        # The result should be a tuple (or other iterable) of actions
        # or a string: 'reset', 'terminate'
        # as defined in the problem description file
        legal_actions.append("reset")
        legal_actions.append("terminate")

        return tuple(legal_actions)

    def result(self, state: str, action: Tuple[Tuple]) -> str:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        state = str_to_dict(state)
        result_state = deepcopy(state)

        if action == "reset":
            return dict_to_str(self.reset_environment(result_state))
        elif action == "terminate":
            return END_OF_GAME_STATE
        else:
            for action_tuple in action:
                result_state = self._execute_action_tuple(result_state, action_tuple)
            return dict_to_str(result_state)  # back into a hashable

    def reset_environment(self, result_state):
        result_state["taxis"] = deepcopy(self.initial["taxis"])
        result_state["passengers"] = deepcopy(self.initial["passengers"])
        return result_state

    def _execute_action_tuple(self, state: dict, action_tuple: Tuple) -> dict:
        """
        input: state dict, and an action tuple like: (“move”, “taxi_name”, (x, y))
        output: the state dict after preforming the action
        """
        actions_possible = MOVE, PICK_UP, DROP_OFF, REFUEL, WAIT = (
            "move",
            "pick up",
            "drop off",
            "refuel",
            "wait",
        )

        action_type = action_tuple[0]
        taxi_name = action_tuple[1]

        result_state = state.copy()

        # check input is legal
        assert (
            action_type in actions_possible
        ), f"{action_type} is not a possible action!"

        if action_type == MOVE:  # (“move”, “taxi_name”, (x, y))
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            # taxi updates:
            result_state["taxis"][taxi_name]["fuel"] -= 1
            result_state["taxis"][taxi_name]["location"] = action_tuple[2]

        elif action_type == PICK_UP:  # (“pick up”, “taxi_name”, “passenger_name”)
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            passenger_name = action_tuple[2]
            # Taxi updates:
            result_state["taxis"][taxi_name]["capacity"] -= 1
            # Passenger updates:
            result_state["passengers"][passenger_name]["location"] = taxi_name

        elif action_type == DROP_OFF:  # (“drop off”, “taxi_name”, “passenger_name”)
            assert (
                len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            passenger_name = action_tuple[2]
            # Taxi updates:
            result_state["taxis"][taxi_name]["capacity"] += 1
            # Passenger updates: passenger location = taxi location
            result_state["passengers"][passenger_name]["location"] = result_state[
                "taxis"
            ][taxi_name]["location"]

        elif action_type == REFUEL:  # ("refuel", "taxi_name")
            assert (
                len(action_tuple) == 2
            ), f"len of action_tuple should be 2: {action_tuple}"
            # taxi updates: fuel = max_fuel
            result_state["taxis"][taxi_name]["fuel"] = self.max_fuel_per_taxi[taxi_name]

        elif action_type == WAIT:  # ("wait", "taxi_name")
            assert (
                len(action_tuple) == 2
            ), f"len of action_tuple should be 2: {action_tuple}"
            pass

        return result_state


class TaxiAgent:
    MAX_PASSENGERS = 2
    PROB_SKIP_CALCULATIONS = 0
    NUM_TOP_ACT = 2
    PROB_TO_ADD_STATE = 1.0

    def __init__(self, initial: dict):
        start = time.time()
        random.seed(42)
        self.initial = initial
        # we shouldn't add anything to the initial state! because we get the state from user in "act" func
        self.max_turns_to_go = initial["turns to go"]
        self.max_fuel_per_taxi = self.get_max_fuel_per_taxi(initial)
        self.max_capacity_per_taxi = self.get_max_capacity_per_taxi(initial)
        self.n_passengers = len(initial["passengers"].keys())
        self.map_num_rows = len(initial["map"])
        self.map_num_cols = len(initial["map"][0])

        self.all_possible_states = []
        self.all_possible_actions = []  # list [a1, a2, ...]
        self.all_possible_actions_for_state = {}  # dict {state: a1, a2, ...}
        self.all_legal_locations_by_passenger = None

        # for Value Iteration
        self.Next_States_Probs = {}
        self.Rewards = {}
        self.Actions_Values = {}
        self.Values = {}
        self.Best_Actions = {}

        self.best_passengers_names = []
        self.passengers_init = self._get_best_passenger_in_init_state()
        self.taxis_init = self._get_best_taxi_in_init_state()

        self.set_all_possible_states()
        self.set_all_possible_actions_and_next_states()
        self.set_rewards_for_actions()
        self.value_iteration_with_dicts()

        print('TaxiAgent initialized in', time.time() - start)

    def act(self, state: dict):
        """
        Get the best action by best_actions dict.
        best actions dict was precalculated by value iteration in the init of agent
        """
        state = deepcopy(state)
        state['passengers'] = self._get_best_pass_dict(state)
        # state['taxis'] = self._get_best_taxi_dict(state)
        t = state["turns to go"]
        state.pop("turns to go")  # remove t from state
        state_str = dict_to_str(state)
        if t == self.max_turns_to_go:    # for debug
            print(f"{t=}", self.Values[t][state_str])
        try:
            best_action = self.Best_Actions[t][state_str]
        except Exception as e:
            print()
        return best_action

    def _get_best_pass_dict(self, state):
        best_pass_dict = dict()
        best_pass_dict[self.best_pass_name] = state['passengers'][self.best_pass_name]
        return best_pass_dict

    def _get_best_taxi_dict(self, state):
        best_taxi_dict = dict()
        best_taxi_dict[self.best_taxi_name] = state['taxis'][self.best_taxi_name]
        return best_taxi_dict

    # ~~~~~~~~~~~~~~~~~~~~~ ALL POSSIBLE STATES

    def set_all_possible_states(self):
        """
        Get all possible permutations of states
        Retruns list of states as json strings.
        [s1, s2, s3, ...]
        states parameters are: optimal, game_map, taxis_option, passengers_option
            (and not including turns to go)
        """
        optimal = self.initial["optimal"]
        game_map = self.initial["map"]
        taxis_init = self.taxis_init
        passengers_init = self.passengers_init
        all_taxis_options = self.get_all_possible_taxis_dicts(taxis_init, game_map)
        all_passengers_options = self.get_all_possible_passengers_dicts(
            passengers_init, taxis_init
        )
        for taxi_name, taxi_dict in self.initial['taxis'].items():
            if taxi_name != self.best_taxi_name:
                for option in all_taxis_options:
                    option[taxi_name] = taxi_dict

        all_possible_states = []
        for taxis_option in all_taxis_options:
            for passengers_option in all_passengers_options:
                # create state dict
                state = {
                    "optimal": optimal,
                    "map": game_map,
                    "taxis": sort_dict(taxis_option),
                    "passengers": passengers_option,
                }
                # checl legal capacity and passenger location
                if self.is_legal_capacity_and_passengers_locations(state):
                    all_possible_states.append(dict_to_str(state))
        all_possible_states.append(END_OF_GAME_STATE)
        self.all_possible_states = all_possible_states

    def _get_best_passenger_in_init_state(self):
        passengers_init = self.initial["passengers"]
        list_lens_goals_lists = [len(p_d['possible_goals']) for p_d in passengers_init.values()]
        list_passes_names = list(passengers_init.keys())
        best_pass_index = np.argmin(list_lens_goals_lists)
        self.best_pass_name = list_passes_names[best_pass_index]

        best_pass_dict = dict()
        best_pass_dict[self.best_pass_name] = passengers_init[self.best_pass_name]
        return best_pass_dict

    def _get_best_taxi_in_init_state(self):
        taxis_init = self.initial["taxis"]
        list_fuel = [p_d['fuel'] for p_d in taxis_init.values()]
        list_taxis_names = list(taxis_init.keys())
        best_taxi_index = np.argmax(list_fuel)
        self.best_taxi_name = list_taxis_names[best_taxi_index]

        best_taxi_dict = dict()
        best_taxi_dict[self.best_taxi_name] = taxis_init[self.best_taxi_name]
        return best_taxi_dict

    def is_legal_capacity_and_passengers_locations(self, state):
        """
        example: num_passengers in taxi   = 3
                 max_capacity of taxi     = 4
                 current_capacity of taxi = 1
                 current_capacity + num_passengers =? max_capacity
                 1 + 3 =? 4 ==> is legal!
        """
        legal = True
        num_passengers_per_taxi = {taxi_name: 0 for taxi_name in state["taxis"].keys()}
        for pass_dict in state["passengers"].values():
            pass_location = pass_dict["location"]
            if pass_location in state["taxis"]:
                num_passengers_per_taxi[pass_location] += 1
        for taxi_name, taxi_details in state["taxis"].items():
            if (
                    taxi_details["capacity"] + num_passengers_per_taxi[taxi_name]
                    != self.max_capacity_per_taxi[taxi_name]
            ):
                legal = False

        # # check also that no 2 taxis in the same location
        # all_taxis_locations = [t['location'] for t in state['taxis'].values()]
        # if len(all_taxis_locations) != len(list(set(all_taxis_locations))):
        #     legal = False

        return legal

    def get_all_possible_taxis_dicts(self, taxis_init: dict, game_map: list) -> list:
        """
        get all possible taxis dicts
        :param dict taxis_init: taxis dict in initial state of game.
        :param list game_map:
        """
        all_options_by_taxi = {}
        # taxi possible locations is the same for all taxis
        possible_locations = self.get_all_legal_taxis_locations(game_map)
        for taxi_name, taxi_details in taxis_init.items():
            all_options_by_taxi[taxi_name] = []
            possible_fuel_vals = range(taxi_details["fuel"] + 1)
            possible_cacpacity_vals = range(taxi_details["capacity"] + 1)
            for location in possible_locations:
                for fuel_value in possible_fuel_vals:
                    for capacity_value in possible_cacpacity_vals:
                        taxi_state = {
                            taxi_name: {
                                "location": location,
                                "fuel": fuel_value,
                                "capacity": capacity_value,
                            }
                        }
                        all_options_by_taxi[taxi_name].append(taxi_state)

        # Get all permutations of taxis dicts (permutation contains one dict of each taxi)
        taxis_options_lists = list(all_options_by_taxi.values())
        if len(taxis_options_lists) == 1:
            all_taxis_permutations = [[i] for i in taxis_options_lists[0]]
        else:
            all_taxis_permutations = list(itertools.product(*taxis_options_lists))

        # create taxis dict from each permutation of dicts
        all_taxis_options = []
        for permutation in all_taxis_permutations:
            taxis_dict = {}
            for i, taxi_name in enumerate(taxis_init.keys()):
                taxis_dict[taxi_name] = permutation[i][taxi_name]
            all_taxis_options.append(taxis_dict)
        return all_taxis_options

    def get_all_legal_taxis_locations(self, game_map):
        """
        taxi location is any location on game_map that is not "I"
        """
        taxis_locations = [
            (i, j) for i in range(self.map_num_rows) for j in range(self.map_num_cols)
        ]
        for x, y in taxis_locations:
            # remove I tiles
            if game_map[x][y] == "I":
                taxis_locations.remove((x, y))
        return taxis_locations

    def get_all_possible_passengers_dicts(self, passengers_init, taxis_init):
        """
        get all possible passengers dicts
        :param dict passengers_init: passengers dict in initial state of game.
        :param dict taxis_init: taxis dict in initial state of game.
        """
        all_options_by_passenger = {}
        # passenger possible location depends on passenger params
        all_legal_locations_by_passenger = self.get_all_legal_locations_by_passenger(
            passengers_init, taxis_init
        )
        for passenger_name, passenger_details in passengers_init.items():
            all_options_by_passenger[passenger_name] = []
            possible_locations = all_legal_locations_by_passenger[passenger_name]
            possible_destinations = passenger_details["possible_goals"]
            # possible_goals and prob_change_goal do not change between states
            possible_goals = passenger_details["possible_goals"]
            prob_change_goal = passenger_details["prob_change_goal"]
            for location in possible_locations:
                for destination in possible_destinations:
                    passenger_state = {
                        passenger_name: {
                            "location": location,
                            "destination": destination,
                            "possible_goals": possible_goals,
                            "prob_change_goal": prob_change_goal,
                        }
                    }
                    all_options_by_passenger[passenger_name].append(passenger_state)

        # Get all permutations of passengers dicts (permutation contains one dict of each passenger)
        passengers_options_lists = list(all_options_by_passenger.values())
        if len(passengers_options_lists) == 1:
            all_passengers_permutations = [[i] for i in passengers_options_lists[0]]
        else:
            all_passengers_permutations = list(
                itertools.product(*passengers_options_lists)
            )

        # create passengers dict from each permutation of dicts
        all_passengers_options = []
        for permutation in all_passengers_permutations:
            passengers_dict = {}
            for i, pass_name in enumerate(passengers_init.keys()):
                passengers_dict[pass_name] = permutation[i][pass_name]
            all_passengers_options.append(passengers_dict)
        return all_passengers_options

    def get_all_legal_locations_by_passenger(self, passengers_init, taxis_init):
        """
        passenger location is one of the following:
            1. the initialized location of this passenger
            2. one of the taxis names (when taxi picked him up)
            3. one of the possible goals of this passenger (when taxi dropped him offß)
        """
        if self.all_legal_locations_by_passenger is not None:
            return self.all_legal_locations_by_passenger

        taxis_names = list(taxis_init.keys())
        all_legal_locations_by_passenger = {}
        for curr_passenger, passengers_dict in passengers_init.items():
            init_location = [tuple(passengers_dict["location"])]
            possible_goals = list(passengers_dict["possible_goals"])
            all_legal_locations_by_passenger[curr_passenger] = set(
                init_location + taxis_names + possible_goals
            )
        return all_legal_locations_by_passenger

    # ~~~~~~~~~~~~~~~~~~~~~ ALL POSSIBLE ACTIONS AND NEXT STATES

    def set_all_possible_actions_and_next_states(self):
        """
        Update:
        @ list: self.all_possible_actions_for_state
        @ list: self.all_possible_actions
        @ dict: self.Next_States_Probs {(state0, action1):
                                            [(state1, prob1), (state2, prob2)]}
                For each possible state, checks the possible next states
                with each action and the probability to get to this next step.
        """
        for state in self.all_possible_states:
            legal_actions = self.actions(state)
            self.all_possible_actions_for_state[state] = legal_actions
            if len(legal_actions) == 0:
                continue
            if state == END_OF_GAME_STATE:
                continue
            for action in legal_actions:
                if action not in self.all_possible_actions:
                    self.all_possible_actions.append(action)
                # find next_states and probs
                if action == "terminate":
                    self.Next_States_Probs[(state, action)] = ()
                    continue
                next_states_with_probs = self.result_with_probs(state, action)
                self.Next_States_Probs[(state, action)] = []
                for next_state, prob in next_states_with_probs.items():
                    self.Next_States_Probs[(state, action)].append((next_state, prob))
    # def get_max_rewards_actions(self, legal_actions, top_num=NUM_TOP_ACT):
    #     list_act_reward_tuples = []
    #     for action in legal_actions:
    #         reward = self._get_reward_for_action(action)
    #         list_act_reward_tuples.append((action, reward))
    #     list_act_reward_tuples.sort(key=lambda x: x[1], reverse=True)
    #
    #     if len(list_act_reward_tuples) <= 3:
    #         list_max_to_return = [x[0] for x in list_act_reward_tuples[:top_num]]
    #
    #     else:
    #         all_wait_action = tuple([("wait", taxi_name) for taxi_name in self.initial["taxis"].keys()])
    #         assert all_wait_action in legal_actions
    #         all_wait_act_reward_tuple = tuple([x for x in list_act_reward_tuples if x[0] == tuple(all_wait_action)][0])
    #         list_act_reward_tuples.remove(all_wait_act_reward_tuple)
    #         list_max_to_return = [x[0] for x in list_act_reward_tuples[:top_num]]
    #
    #     return list_max_to_return

    def result_with_probs(self, state, action):
        """
        input: state, action
        output: next possible states and the probability for each one
        method:
            1.  Use the deterministic result function to get the next "regular" state.
            2.  Get all possible new states with possible changes of destinations
                2a. Get new_state prob and possible destinations
                    where the subset passengers may change their destination
                2b. Get all destinations permutations
                    (passengers have few possible goals to change to)
                2c. For each permutation get new state with updated destinations

        """
        possible_next_state_probs = {}
        # 1. Use the deterministic result function to get the next "regular" state.
        result_state_str = self.result(state, action)

        # 2. Get all possible new states with possible changes of destinations
        if action == "reset":
            reset_prob = 1  # we saw in check.py that they are not doing rechoice
            possible_next_state_probs[result_state_str] = reset_prob
            return possible_next_state_probs

        result_state = str_to_dict(result_state_str)
        all_passengers_subsets = self.get_all_passengers_subsets(result_state)

        for pass_subset_names in all_passengers_subsets:
            # 2a. Get new_state prob and possible destinations lists
            #     where the subset passengers may change their destination
            new_state_prob, dest_lists = self.get_prob_and_dest_lists(
                result_state, pass_subset_names
            )

            # 2b. Get all destinations permutations
            all_dest_permu = self.get_all_dest_permutations(dest_lists)

            # 2c. For each permutation get new state with updated destinations
            #     and update possible_next_state_probs dict
            for dest_permu in all_dest_permu:
                new_state = self.get_new_state(result_state, dest_permu)
                possible_next_state_probs = self.add_state_and_prob(
                    new_state, new_state_prob, possible_next_state_probs
                )

        return possible_next_state_probs

    def get_prob_and_dest_lists(self, result_state, pass_subset_names):
        new_state_prob = 1  # init prob
        dest_lists = []
        for pass_name, pass_details in result_state["passengers"].items():
            if pass_name in pass_subset_names:  # passenger rechoice destination
                possible_goals = list(pass_details["possible_goals"])
                n_goals = float(len(possible_goals))
                new_state_prob *= (pass_details["prob_change_goal"]) / n_goals
            else:
                possible_goals = list([pass_details["destination"]])
                new_state_prob *= 1.0 - pass_details["prob_change_goal"]
            dest_lists.append(possible_goals)
        return new_state_prob, dest_lists

    def get_all_passengers_subsets(self, result_state):
        if self.n_passengers == 1:
            all_passengers_subsets = list(result_state["passengers"].keys())
        else:
            all_passengers_subsets = powerset(result_state["passengers"].keys())
        all_passengers_subsets.append(
            [()]
        )  # no passenger is rechoicing his destination
        return all_passengers_subsets

    def get_all_dest_permutations(self, possible_dest_lists):
        if len(possible_dest_lists) <= 1:
            all_dest_permu = possible_dest_lists[0]
        else:
            all_dest_permu = list(itertools.product(*possible_dest_lists))
        return all_dest_permu

    def get_new_state(self, result_state, dest_permu):
        """
        @ dest_permu: destinations permutation
            example [(0,0), (2,1), (3,3)] permutaion of destinations
                for [pass1, pass2, pass3]
        """
        new_state = deepcopy(result_state)
        for pass_name in new_state["passengers"].keys():
            new_state["passengers"][pass_name]["destination"] = dest_permu
        return new_state

    def add_state_and_prob(self, new_state, new_state_prob, possible_next_state_probs):
        new_state_str = dict_to_str(new_state)
        if new_state_str in possible_next_state_probs.keys():  # sum probs
            existing_prob = possible_next_state_probs[new_state_str]
            additional_prob = new_state_prob
            possible_next_state_probs[new_state_str] = existing_prob + additional_prob
        else:
            possible_next_state_probs[new_state_str] = new_state_prob
        return possible_next_state_probs

    # ~~~~~~~~~~~~~~~~~~~~~ REWARDS

    def set_rewards_for_actions(self, list_actions=None):
        """
        For each action set the reward.
        """
        if list_actions is None:
            list_actions = self.all_possible_actions
        for action in list_actions:
            if action == "reset":
                reward = -RESET_PENALTY
            elif action == "terminate":
                reward = 0
            else:
                reward = 0
                for atomic_action in action:
                    if atomic_action[0] == "drop off":
                        reward += DROP_IN_DESTINATION_REWARD
                    elif atomic_action[0] == "refuel":
                        reward -= REFUEL_PENALTY
            self.Rewards[action] = reward

    def _get_reward_for_action(self, action):
        if action == "reset":
            reward = -RESET_PENALTY
        elif action == "terminate":
            reward = 0
        else:
            reward = 0
            for atomic_action in action:
                if atomic_action[0] == "drop off":
                    reward += DROP_IN_DESTINATION_REWARD
                elif atomic_action[0] == "refuel":
                    reward -= REFUEL_PENALTY
        return reward

    # ~~~~~~~~~~~~~~~~~~~~~ VI

    def value_iteration_with_dicts(self):
        """
        Get the best action for each step t
        from t = 0 to t = max_t.
        Update self.Best_Actions dict.
        Method: running value iteration algorithm.
        Using:
        @ Next_States_Probs: dict {(state0, action1):
                                    [(state1, prob1), (state2, prob2)]}
        @ Rewards: dict {(action): reward}  # depends only on action!!
        @ Actions_Values    {t:
                                {state0:
                                    {(action1): value,
                                     (action2): value}
                                {state1}:
                                    {(action1): value,
                                     (action2): value}
                            }
        @ Best_Actions:     {t:
                                {state0: best_action,
                                 state1: best_action}
                            }
        @ Values            {t:
                                {state0: best_value,
                                 state1: best_value}
                            }
        """
        for t in range(self.max_turns_to_go + 1):
            self.Values[t] = {}
            self.Actions_Values[t] = {}
            self.Best_Actions[t] = {}
            if t == 0:
                # End of game - no action is possible --> value is 0 for all states.
                for state in self.all_possible_states:
                    self.Values[t][state] = 0
            else:
                for state in self.all_possible_states:
                    if state == END_OF_GAME_STATE:
                        self.Values[t][state] = 0
                        continue
                    self.Actions_Values[t][state] = {}
                    # Update - Actions_Values dict
                    for action in self.all_possible_actions_for_state[state]:
                        # V(t, s) = max_over_a{
                        #              sum_over_next_s[prob(next_s, a)* V(t-1, next_s)] + Reward(a)}
                        value_of_action = self.Rewards[action]  # init
                        for next_state, prob in self.Next_States_Probs[(state, action)]:
                            try:
                                value_of_action += prob * self.Values[t - 1][next_state]
                            except:
                                print()
                        self.Actions_Values[t][state][action] = value_of_action
                    # Find - best action and best value
                    possible_actions = list(self.Actions_Values[t][state].keys())
                    possible_values = list(self.Actions_Values[t][state].values())
                    best_value = max(possible_values)
                    best_action = possible_actions[np.argmax(possible_values)]
                    # Update  - Values and Best_Actions dicts
                    self.Values[t][state] = best_value
                    self.Best_Actions[t][state] = best_action

    # ~~~~~~~~~~~~~~~~~~~~~

    def get_max_fuel_per_taxi(self, initial):
        max_fuel_per_taxi = {}
        for taxi_name, taxi_dict in initial["taxis"].items():
            max_fuel_per_taxi[taxi_name] = taxi_dict["fuel"]
        return max_fuel_per_taxi

    def get_max_capacity_per_taxi(self, initial):
        max_capacity_per_taxi = {}
        for taxi_name, taxi_dict in initial["taxis"].items():
            max_capacity_per_taxi[taxi_name] = taxi_dict["capacity"]
        return max_capacity_per_taxi

    def generate_locations(self, state: dict) -> dict:
        # get new locations by:
        # current location + one step in legal orientation (EAST, NORTH, WEST, SOUTH)
        possible_locations_by_taxi = dict()
        for taxi_name, taxi_dict in state["taxis"].items():
            curr_location = taxi_dict["location"]
            possible_locations = [
                vector_add(curr_location, orient) for orient in orientations
            ]
            possible_locations_by_taxi[taxi_name] = possible_locations

        return possible_locations_by_taxi

    def get_legal_moves_on_map(self, state: dict) -> dict:
        legal_locations_by_taxi = {}
        possible_locations_by_taxi = self.generate_locations(state)
        for taxi_name, taxi_dict in state["taxis"].items():
            # 1. check fuel > 0
            legal_locations = []
            if taxi_dict["fuel"] > 0:
                possible_locations = possible_locations_by_taxi[taxi_name]
                for new_location in possible_locations:
                    x, y = new_location
                    # 2. check that the taxi doesn't get out of the map
                    # 3. check that the taxi is on a passable tile
                    if (0 <= x < self.map_num_rows) and (0 <= y < self.map_num_cols):
                        if state["map"][x][y] != "I":
                            legal_locations.append(new_location)
            legal_locations_by_taxi[taxi_name] = legal_locations
        return legal_locations_by_taxi

    def get_legal_refuel(self, state: dict) -> dict:
        # Refueling can be performed only at gas stations
        legal_refuels_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            map_matrix = state["map"]
            x, y = taxi_dict["location"]  # current location of taxi
            # check that the location on map is "G"
            legal_refuel = map_matrix[x][y] == "G"  # bool
            legal_refuels_by_taxi[taxi_name] = legal_refuel
        return legal_refuels_by_taxi

    def get_legal_pick_up(self, state: dict) -> dict:
        # Pick up passengers if they are on the same tile as the taxi.
        legal_pickups_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            capacity = taxi_dict["capacity"]
            legal_pickups = []
            # The number of passengers in the taxi has to be < taxi’s capacity.
            if capacity > 0:
                for passenger_name, passenger_dict in state["passengers"].items():
                    # check that location of taxi is the same as location of the passenger
                    # and check that the location of the passenger is not his destination
                    if (taxi_dict["location"] == passenger_dict["location"]) & (
                            passenger_dict["location"] != passenger_dict["destination"]
                    ):
                        legal_pickups.append(passenger_name)
            legal_pickups_by_taxi[taxi_name] = legal_pickups
        return legal_pickups_by_taxi

    def get_legal_drop_off(self, state: dict) -> dict:
        # The passenger can only be dropped off on his destination tile
        # and will refuse to leave the vehicle otherwise.
        legal_drop_offs_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            legal_drop_offs = []
            # go over the passengers that's on the curr taxi
            for passenger_name, passenger_dict in state["passengers"].items():
                if passenger_dict["location"] == taxi_name:
                    # check that location of taxi is the same as destination of the passenger
                    if taxi_dict["location"] == passenger_dict["destination"]:
                        legal_drop_offs.append(passenger_name)
            legal_drop_offs_by_taxi[taxi_name] = legal_drop_offs
        return legal_drop_offs_by_taxi

    def actions(self, state: str) -> Tuple[Tuple[Tuple]]:
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        # -----------------------------------------------------------------
        # Atomic Actions: ["move", "pick_up", "drop_off", "refuel", "wait"]
        # explicit syntax:
        # (“move”, “taxi_name”, (x, y))
        # (“pick up”, “taxi_name”, “passenger_name”
        # (“drop off”, “taxi_name”, “passenger_name”)
        # ("refuel", "taxi_name")
        # ("wait", "taxi_name")

        # Full Action - a tuple with action for each taxi
        # Example: ((“move”, “taxi 1”, (1, 2)),
        #           (“wait”, “taxi 2”),
        #           (“pick up”, “very_fancy_taxi”, “Yossi”))
        # -----------------------------------------------------------------

        if state == END_OF_GAME_STATE:
            legal_actions = tuple()
            return legal_actions
        state_original = str_to_dict(state)
        state = str_to_dict(state)

        state['taxis'] = {self.best_taxi_name: state['taxis'][self.best_taxi_name]}
        state['passengers'] = {self.best_pass_name: state['passengers'][self.best_pass_name]}
        # For each taxi get Possible Atomic Actions

        legal_locations_by_taxi = self.get_legal_moves_on_map(
            state
        )  # DICT[taxi_name: list of (x,y) locations]
        legal_pickups_by_taxi = self.get_legal_pick_up(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_drop_offs_by_taxi = self.get_legal_drop_off(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_refuels_by_taxi = self.get_legal_refuel(
            state
        )  # DICT[taxi_name: True / False]

        # -----------------------------------------------------------------
        # Get Atomic Actions with right syntax
        atomic_actions_lists = []
        for taxi_name in self.initial["taxis"].keys():
            if taxi_name == self.best_taxi_name:
                atomic_actions = [("wait", taxi_name)]
                for location in legal_locations_by_taxi[taxi_name]:
                    atomic_actions.append(("move", taxi_name, location))
                for passenger_name in legal_pickups_by_taxi[taxi_name]:
                    atomic_actions.append(("pick up", taxi_name, passenger_name))
                for passenger_name in legal_drop_offs_by_taxi[taxi_name]:
                    atomic_actions.append(("drop off", taxi_name, passenger_name))
                if legal_refuels_by_taxi[taxi_name]:
                    atomic_actions.append(("refuel", taxi_name))
                atomic_actions_lists.append(atomic_actions)
            else:
                atomic_actions = [("wait", taxi_name)]
                atomic_actions_lists.append(atomic_actions)

        # -----------------------------------------------------------------
        # Get Actions - all permutations of atomic actions
        actions = list(itertools.product(*atomic_actions_lists))

        # -----------------------------------------------------------------
        # For each action - Check That Taxis Don't Clash with each other
        #   == not going to the same location (therefore cannot pickup the same passenger)
        if len(state_original["taxis"]) > 1:
            legal_actions = []
            for action in actions:
                taxis_next_locations = []
                for atomic_action in action:  # TODO: NOTE changed from atomic_actions_lists to action
                    action_type = atomic_action[0]
                    taxi_name = atomic_action[1]
                    taxi_curr_location = state_original["taxis"][taxi_name]["location"]
                    if action_type == "move":
                        taxi_next_location = atomic_action[2]
                    else:
                        taxi_next_location = taxi_curr_location
                    taxis_next_locations.append(taxi_next_location)
                # check if there are 2 taxis in the same location
                legal_action = len(set(taxis_next_locations)) == len(state_original["taxis"])
                if legal_action:
                    legal_actions.append(action)
        else:  # n_taxis == 1 --> no clashing between taxis
            legal_actions = actions

        # -----------------------------------------------------------------
        # The result should be a tuple (or other iterable) of actions
        # or a string: 'reset', 'terminate'
        # as defined in the problem description file
        legal_actions.append("reset")
        legal_actions.append("terminate")

        return tuple(legal_actions)

    def result(self, state: str, action: Tuple[Tuple]) -> str:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        state = str_to_dict(state)
        result_state = deepcopy(state)

        if action == "reset":
            return dict_to_str(self.reset_environment(result_state))
        elif action == "terminate":
            return END_OF_GAME_STATE
        else:
            for action_tuple in action:
                result_state = self._execute_action_tuple(result_state, action_tuple)
            return dict_to_str(result_state)  # back into a hashable

    def reset_environment(self, result_state):
        result_state["taxis"] = deepcopy(self.initial["taxis"])
        result_state["passengers"] = deepcopy(self.passengers_init)
        return result_state

    def _execute_action_tuple(self, state: dict, action_tuple: Tuple) -> dict:
        """
        input: state dict, and an action tuple like: (“move”, “taxi_name”, (x, y))
        output: the state dict after preforming the action
        """
        actions_possible = MOVE, PICK_UP, DROP_OFF, REFUEL, WAIT = (
            "move",
            "pick up",
            "drop off",
            "refuel",
            "wait",
        )

        action_type = action_tuple[0]
        taxi_name = action_tuple[1]

        result_state = state.copy()

        # check input is legal
        assert (
                action_type in actions_possible
        ), f"{action_type} is not a possible action!"

        if action_type == MOVE:  # (“move”, “taxi_name”, (x, y))
            assert (
                    len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            # taxi updates:
            result_state["taxis"][taxi_name]["fuel"] -= 1
            result_state["taxis"][taxi_name]["location"] = action_tuple[2]

        elif action_type == PICK_UP:  # (“pick up”, “taxi_name”, “passenger_name”)
            assert (
                    len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            passenger_name = action_tuple[2]
            # Taxi updates:
            result_state["taxis"][taxi_name]["capacity"] -= 1
            # Passenger updates:
            result_state["passengers"][passenger_name]["location"] = taxi_name

        elif action_type == DROP_OFF:  # (“drop off”, “taxi_name”, “passenger_name”)
            assert (
                    len(action_tuple) == 3
            ), f"len of action_tuple should be 3: {action_tuple}"
            passenger_name = action_tuple[2]
            # Taxi updates:
            result_state["taxis"][taxi_name]["capacity"] += 1
            # Passenger updates: passenger location = taxi location
            result_state["passengers"][passenger_name]["location"] = result_state[
                "taxis"
            ][taxi_name]["location"]

        elif action_type == REFUEL:  # ("refuel", "taxi_name")
            assert (
                    len(action_tuple) == 2
            ), f"len of action_tuple should be 2: {action_tuple}"
            # taxi updates: fuel = max_fuel
            result_state["taxis"][taxi_name]["fuel"] = self.max_fuel_per_taxi[taxi_name]

        elif action_type == WAIT:  # ("wait", "taxi_name")
            assert (
                    len(action_tuple) == 2
            ), f"len of action_tuple should be 2: {action_tuple}"
            pass

        return result_state


class TaxiAgentRL:

    N = 10       # depth to calc q
    LR = 0.01    # learning rate
    DF = 0.8     # discount factor
    MAX_ACTIONS = 3

    def __init__(self, initial: dict):
        start = time.time()
        random.seed(42)

        self.initial = initial
        self.map_num_rows = len(initial["map"])
        self.map_num_cols = len(initial["map"][0])
        self.n_passengers = len(initial["passengers"].keys())
        self.max_turns_to_go = initial["turns to go"]

        self.states_list = []
        self.actions_for_state_list = {}  # dict {state: a1, a2, ...}
        self.states_info_dict = {}
        self.unvisited_states = []

        # self.Next_States_Probs = {}
        self.Rewards = {}
        self.State_Action_Values = {}
        # self.Values = {}
        self.Best_Actions = {}

        self.RL_with_n_step_sarsa_algorithm()
        self.set_best_actions()

        print('TaxiAgent initialized in', time.time() - start)

    def act(self, state: dict):
        """
        Get the best action by best_actions dict.
        best actions dict was precalculated by value iteration in the init of agent
        """
        state_str = dict_to_str(state)
        if state_str in self.Best_Actions:
            best_action = self.Best_Actions[state_str]
        else:
            actions = self.actions(state_str)
            best_action = self.get_max_rewards_actions(actions, top_num=1)[0]
        return best_action

    def RL_with_n_step_sarsa_algorithm(self):
        """ hybrid method: Monte carlo + TD learning """
        start = time.time()
        random.seed(42)
        lr = TaxiAgent.LR
        state_str = dict_to_str(self.initial)
        actions = self.actions(state_str)
        self.states_list.append(state_str)
        self.states_info_dict[state_str] = {"actions": actions, "n_visited": 0, "x_mean_reward": 0, "alg_score": -1}
        self.unvisited_states.append(state_str)
        t = 0
        while time.time() - start < 280:
            t += 1
            # =========== selection ===========
            state_str = self.select(start_time=start)

            # =========== expansion + simulation ===========
            curr_state = state_str
            q, first_act = self.calc_q(curr_state)

            # =========== update ===========
            # approximate value func
            if state_str not in self.State_Action_Values.keys():
                print()
            Q_s_a = self.State_Action_Values[state_str][first_act]
            self.State_Action_Values[state_str][first_act] = Q_s_a + lr * (q - Q_s_a)

            # update self.states_info_dict
            self.states_info_dict[state_str]["n_visited"] += 1
            n = self.states_info_dict[state_str]["n_visited"]
            curr_mean = self.states_info_dict[state_str]["x_mean_reward"]
            x = self.states_info_dict[state_str]["x_mean_reward"] = curr_mean + (1 / (n + 1)) * (q - curr_mean)
            self.states_info_dict[state_str]["alg_score"] = x + np.sqrt(2 * np.log(t) / n)

    def select(self, start_time):
        if time.time() - start_time < 8:
            state_str = self.explore()
        else:
            if random.random() < 0.99:
                state_str = self.exploit()
            else:
                state_str = self.explore()

        return state_str

    def explore(self):
        if len(self.unvisited_states) > 0:
            return self.unvisited_states.pop(0)
        else:
            return self.exploit()

    def exploit(self):
        # arg max { x_j + sqrt(2*log(t)/n_j) }
        alg_scores_list = [x['alg_score'] for x in self.states_info_dict.values()]
        state_index = np.argmax(alg_scores_list)
        state_str = list(self.states_info_dict.keys())[state_index]
        return state_str

    def calc_q(self, curr_state):
        discount_f = TaxiAgent.DF
        q = 0.0
        first_act = ()
        for depth in range(TaxiAgent.N):
            if curr_state == END_OF_GAME_STATE \
                    or (str_to_dict(curr_state)["turns to go"] / self.max_turns_to_go) < 0.5:
                break

            # ~~~~~~~~ expend
            if curr_state in self.states_info_dict.keys():
                actions = self.states_info_dict[curr_state]['actions']
            else:
                actions = self.actions(curr_state)
                # actions = self.get_max_rewards_actions(actions)
                self.states_info_dict[curr_state] = {
                    "actions": actions, "n_visited": 0, "x_mean_reward": 0, "alg_score": -1
                }

            # ~~~~~~~~ simulate
            # choose action in total random
            action = random.choice(actions)
            # init dict State_Action_Values
            if curr_state not in self.State_Action_Values:
                self.State_Action_Values[curr_state] = {}
            if action not in self.State_Action_Values[curr_state].keys():
                self.State_Action_Values[curr_state][action] = 0

            if curr_state not in self.actions_for_state_list:
                self.actions_for_state_list[curr_state] = []

            self.actions_for_state_list[curr_state].append(action)

            # add discounted reward
            if action not in self.Rewards.keys():
                self.Rewards[action] = self._get_reward_for_action(action)
            q += (discount_f ** depth) * self.Rewards[action]

            # move to next state with chosen action
            curr_state, other_possible_next_states = self.result(curr_state, action)
            self.unvisited_states += other_possible_next_states   # TODO check
            self.unvisited_states = list(set(self.unvisited_states))

            if depth == 0:
                first_act = action

        return q, first_act

    def get_max_rewards_actions(self, legal_actions, top_num=MAX_ACTIONS):
        if len(legal_actions) <= top_num:
            return legal_actions
        list_act_reward_tuples = []
        for action in legal_actions:
            reward = self._get_reward_for_action(action)
            list_act_reward_tuples.append((action, reward))
        list_act_reward_tuples.sort(key=lambda x: x[1], reverse=True)
        list_max_to_return = [x[0] for x in list_act_reward_tuples[:top_num]]
        return list_max_to_return

    def set_best_actions(self):
        for state in self.states_list:
            # Find - best action and best value
            possible_actions = list(self.State_Action_Values[state].keys())
            possible_values = list(self.State_Action_Values[state].values())
            best_action = possible_actions[np.argmax(possible_values)]
            # Update  - Values and Best_Actions dicts
            self.Best_Actions[state] = best_action

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_reward_for_action(self, action):
        if action == "reset":
            reward = -RESET_PENALTY
        elif action == "terminate":
            reward = 0
        else:
            reward = 0
            for atomic_action in action:
                if atomic_action[0] == "drop off":
                    reward += DROP_IN_DESTINATION_REWARD
                elif atomic_action[0] == "refuel":
                    reward -= REFUEL_PENALTY
        return reward

    def generate_locations(self, state: dict) -> dict:
        # get new locations by:
        # current location + one step in legal orientation (EAST, NORTH, WEST, SOUTH)
        possible_locations_by_taxi = dict()
        for taxi_name, taxi_dict in state["taxis"].items():
            curr_location = taxi_dict["location"]
            possible_locations = [
                vector_add(curr_location, orient) for orient in orientations
            ]
            possible_locations_by_taxi[taxi_name] = possible_locations

        return possible_locations_by_taxi

    def get_legal_moves_on_map(self, state: dict) -> dict:
        legal_locations_by_taxi = {}
        possible_locations_by_taxi = self.generate_locations(state)
        for taxi_name, taxi_dict in state["taxis"].items():
            # 1. check fuel > 0
            legal_locations = []
            if taxi_dict["fuel"] > 0:
                possible_locations = possible_locations_by_taxi[taxi_name]
                for new_location in possible_locations:
                    x, y = new_location
                    # 2. check that the taxi doesn't get out of the map
                    # 3. check that the taxi is on a passable tile
                    if (0 <= x < self.map_num_rows) and (0 <= y < self.map_num_cols):
                        if state["map"][x][y] != "I":
                            legal_locations.append(new_location)
            legal_locations_by_taxi[taxi_name] = legal_locations
        return legal_locations_by_taxi

    def get_legal_refuel(self, state: dict) -> dict:
        # Refueling can be performed only at gas stations
        legal_refuels_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            map_matrix = state["map"]
            x, y = taxi_dict["location"]  # current location of taxi
            # check that the location on map is "G"
            legal_refuel = map_matrix[x][y] == "G"  # bool
            legal_refuels_by_taxi[taxi_name] = legal_refuel
        return legal_refuels_by_taxi

    def get_legal_pick_up(self, state: dict) -> dict:
        # Pick up passengers if they are on the same tile as the taxi.
        legal_pickups_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            capacity = taxi_dict["capacity"]
            legal_pickups = []
            # The number of passengers in the taxi has to be < taxi’s capacity.
            if capacity > 0:
                for passenger_name, passenger_dict in state["passengers"].items():
                    # check that location of taxi is the same as location of the passenger
                    # and check that the location of the passenger is not his destination
                    if (taxi_dict["location"] == passenger_dict["location"]) & (
                        passenger_dict["location"] != passenger_dict["destination"]
                    ):
                        legal_pickups.append(passenger_name)
            legal_pickups_by_taxi[taxi_name] = legal_pickups
        return legal_pickups_by_taxi

    def get_legal_drop_off(self, state: dict) -> dict:
        # The passenger can only be dropped off on his destination tile
        # and will refuse to leave the vehicle otherwise.
        legal_drop_offs_by_taxi = {}
        for taxi_name, taxi_dict in state["taxis"].items():
            legal_drop_offs = []
            # go over the passengers that's on the curr taxi
            for passenger_name, passenger_dict in state["passengers"].items():
                if passenger_dict["location"] == taxi_name:
                    # check that location of taxi is the same as destination of the passenger
                    if taxi_dict["location"] == passenger_dict["destination"]:
                        legal_drop_offs.append(passenger_name)
            legal_drop_offs_by_taxi[taxi_name] = legal_drop_offs
        return legal_drop_offs_by_taxi

    def actions(self, state: str) -> Tuple[Tuple[Tuple]]:
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        # -----------------------------------------------------------------
        # Atomic Actions: ["move", "pick_up", "drop_off", "refuel", "wait"]
        # explicit syntax:
        # (“move”, “taxi_name”, (x, y))
        # (“pick up”, “taxi_name”, “passenger_name”
        # (“drop off”, “taxi_name”, “passenger_name”)
        # ("refuel", "taxi_name")
        # ("wait", "taxi_name")

        # Full Action - a tuple with action for each taxi
        # Example: ((“move”, “taxi 1”, (1, 2)),
        #           (“wait”, “taxi 2”),
        #           (“pick up”, “very_fancy_taxi”, “Yossi”))
        # -----------------------------------------------------------------

        if state == END_OF_GAME_STATE:
            legal_actions = tuple()
            return legal_actions
        state = str_to_dict(state)

        # For each taxi get Possible Atomic Actions

        legal_locations_by_taxi = self.get_legal_moves_on_map(
            state
        )  # DICT[taxi_name: list of (x,y) locations]
        legal_pickups_by_taxi = self.get_legal_pick_up(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_drop_offs_by_taxi = self.get_legal_drop_off(
            state
        )  # DICT[taxi_name: list of passengers names]
        legal_refuels_by_taxi = self.get_legal_refuel(
            state
        )  # DICT[taxi_name: True / False]

        # -----------------------------------------------------------------
        # Get Atomic Actions with right syntax
        atomic_actions_lists = []
        for taxi_name in state["taxis"].keys():
            atomic_actions = [("wait", taxi_name)]
            for location in legal_locations_by_taxi[taxi_name]:
                atomic_actions.append(("move", taxi_name, location))
            for passenger_name in legal_pickups_by_taxi[taxi_name]:
                atomic_actions.append(("pick up", taxi_name, passenger_name))
            for passenger_name in legal_drop_offs_by_taxi[taxi_name]:
                atomic_actions.append(("drop off", taxi_name, passenger_name))
            if legal_refuels_by_taxi[taxi_name]:
                atomic_actions.append(("refuel", taxi_name))
            atomic_actions_lists.append(atomic_actions)

        # -----------------------------------------------------------------
        # Get Actions - all permutations of atomic actions
        actions = list(itertools.product(*atomic_actions_lists))
        # all_wait_action = tuple(
        #     [("wait", taxi_name) for taxi_name in state["taxis"].keys()]
        # )
        # # assert all_wait_action in actions
        # # actions.remove(all_wait_action)

        # -----------------------------------------------------------------
        # For each action - Check That Taxis Don't Clash with each other
        #   == not going to the same location (therefore cannot pickup the same passenger)
        if len(state["taxis"]) > 1:
            legal_actions = []
            for action in actions:
                taxis_next_locations = []
                for (
                    atomic_action
                ) in action:  # TODO: NOTE changed from atomic_actions_lists to action
                    action_type = atomic_action[0]
                    taxi_name = atomic_action[1]
                    taxi_curr_location = state["taxis"][taxi_name]["location"]
                    if action_type == "move":
                        taxi_next_location = atomic_action[2]
                    else:
                        taxi_next_location = taxi_curr_location
                    taxis_next_locations.append(taxi_next_location)
                # check if there are 2 taxis in the same location
                legal_action = len(set(taxis_next_locations)) == len(state["taxis"])
                if legal_action:
                    legal_actions.append(action)
        else:  # n_taxis == 1 --> no clashing between taxis
            legal_actions = actions

        # -----------------------------------------------------------------
        # The result should be a tuple (or other iterable) of actions
        # or a string: 'reset', 'terminate'
        # as defined in the problem description file
        legal_actions.append("reset")
        legal_actions.append("terminate")

        return tuple(legal_actions)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def result(self, state: str, action):
        """"
        update the state according to the action
        """
        state = str_to_dict(state)
        res_state = self.apply(state, action)
        other_possible_next_states = []
        if res_state == END_OF_GAME_STATE:
            res_state = dict_to_str(res_state)
            return res_state, other_possible_next_states
        if action != "reset":
            res_state = self.environment_step(res_state)
            other_possible_next_states = self.get_possible_next_states(res_state)
            other_possible_next_states.remove(dict_to_str(res_state))
        res_state = dict_to_str(res_state)
        return res_state, other_possible_next_states

    def apply(self, state: dict, action):
        """
        apply the action to the state
        """
        if action == "terminate" or state["turns to go"] == 0:
            return END_OF_GAME_STATE
        if action == "reset":
            return self.reset_environment(state)
        return_state = deepcopy(state)
        for atomic_action in action:
            return_state = self.apply_atomic_action(return_state, atomic_action)
        return return_state

    def apply_atomic_action(self, state, atomic_action):
        """
        apply an atomic action to the state
        """
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            state['taxis'][taxi_name]['location'] = atomic_action[2]
            state['taxis'][taxi_name]['fuel'] -= 1
            return state
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            state['taxis'][taxi_name]['capacity'] -= 1
            state['passengers'][passenger_name]['location'] = taxi_name
            return state
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            state['passengers'][passenger_name]['location'] = state['taxis'][taxi_name]['location']
            state['taxis'][taxi_name]['capacity'] += 1
            return state
        elif atomic_action[0] == 'refuel':
            state['taxis'][taxi_name]['fuel'] = self.initial['taxis'][taxi_name]['fuel']
            return state
        elif atomic_action[0] == 'wait':
            return state
        else:
            raise NotImplemented

    def environment_step(self, state: dict):
        """
        update the state of environment randomly
        """
        for p in state['passengers']:
            passenger_stats = state['passengers'][p]
            if random.random() < passenger_stats['prob_change_goal']:
                # change destination
                passenger_stats['destination'] = random.choice(passenger_stats['possible_goals'])
                # possible next states
        state["turns to go"] -= 1
        return state

    def get_possible_next_states(self, state: dict):
        all_passengers_subsets = self.get_all_passengers_subsets(state)
        other_possible_next_states = []

        for pass_subset_names in all_passengers_subsets:
            # Get the possible destinations lists where the subset passengers may change their destination
            dest_lists = []
            for pass_name, pass_details in state["passengers"].items():
                if pass_name in pass_subset_names:  # passenger rechoice destination
                    possible_goals = list(pass_details["possible_goals"])
                else:
                    possible_goals = list([pass_details["destination"]])
                dest_lists.append(possible_goals)

            # Get all destinations permutations
            all_dest_permu = self.get_all_dest_permutations(dest_lists)

            # For each permutation get new state with updated destinations
            for dest_permu in all_dest_permu:
                new_state = self.get_new_state(state, dest_permu)
                other_possible_next_states.append(new_state)

        other_possible_next_states_strs = [dict_to_str(d) for d in other_possible_next_states]
        return other_possible_next_states_strs

    def get_all_dest_permutations(self, possible_dest_lists):
        if len(possible_dest_lists) <= 1:
            all_dest_permu = possible_dest_lists[0]
        else:
            all_dest_permu = list(itertools.product(*possible_dest_lists))
        return all_dest_permu

    def get_all_passengers_subsets(self, result_state):
        all_passengers_subsets = []
        if self.n_passengers == 1:
            all_passengers_subsets = list(result_state["passengers"].keys())

        else:
            all_passengers_subsets = powerset(result_state["passengers"].keys())
        return all_passengers_subsets

    def get_new_state(self, result_state, dest_permu):
        """
        @ dest_permu: destinations permutation
            example [(0,0), (2,1), (3,3)] permutaion of destinations
                for [pass1, pass2, pass3]
        """
        new_state = deepcopy(result_state)
        if self.n_passengers == 1:
            for pass_name in new_state["passengers"].keys():
                new_state["passengers"][pass_name]["destination"] = dest_permu
        else:
            for i, pass_name in enumerate(new_state["passengers"].keys()):
                new_state["passengers"][pass_name]["destination"] = dest_permu[i]
        return new_state

    def reset_environment(self, state):
        """
        reset the state of the environment
        """
        state["taxis"] = deepcopy(self.initial["taxis"])
        state["passengers"] = deepcopy(self.initial["passengers"])
        state["turns to go"] -= 1
        return state


def dict_to_str(d: dict) -> str:
    d_str = str(d)
    return d_str


def str_to_dict(s: str) -> dict:
    j_dict = ast.literal_eval(s)
    return j_dict


def sort_dict(d: dict) -> dict:
    sorted_keys = sorted(d)
    return {key: d[key] for key in sorted_keys}

if __name__ == "__main__":
    state = {
        "optimal": True,
        "map": [["P", "P", "P"], ["P", "G", "P"], ["P", "P", "P"]],
        "taxis": {"taxi 1": {"location": (0, 0), "fuel": 10, "capacity": 1}},
        "passengers": {
            "Dana": {
                "location": (2, 2),
                "destination": (0, 0),
                "possible_goals": ((0, 0), (2, 2)),
                "prob_change_goal": 0.1,
            }
        },
        "turns to go": 100,
    }

    TaxiAgent(initial=state)

    # exit()
    # for i, s in enumerate(additional_inputs):
    #     print("---------------", i+1, "---------------")
    #     agent = TaxiAgent(initial=s)
