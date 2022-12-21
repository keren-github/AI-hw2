import ast
import itertools
from copy import deepcopy
from typing import Tuple

# import networkx as nx
import numpy as np

from utils import orientations, powerset, vector_add

ids = ["316375872", "206014482"]

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1


class OptimalTaxiAgent:
    def __init__(self, initial: dict):
        self.initial = initial
        # we shouldnt add anything to th states! because we get the state from user in "act" func
        self.max_turns_to_go = initial["turns to go"]
        self.max_fuel_per_taxi = self.get_max_fuel_per_taxi(initial)

        self.all_possible_states = self.set_all_possible_states()
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
        self.value_iteration_with_dicts()

    def act(self, state: dict):
        """
        Get the best action by best_actions dict.
        best actions dict was precalculated by value iteration in the init of agent
        """
        t = state["turns to go"]
        state_without_t = state.pop("turns to go")
        best_action = self.best_actions[(state_without_t, t)]
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
        all_taxis_permutations = self.get_all_possible_taxis_dicts(taxis_init, game_map)
        all_passengers_permutations = self.get_all_possible_passengers_dicts(
            passengers_init, taxis_init
        )

        all_possible_states = []
        for taxis_option in all_taxis_permutations:
            if len(taxis_option) == 1:
                taxis_option = taxis_option[0]
            for passengers_option in all_passengers_permutations:
                if len(passengers_option) == 1:
                    passengers_option = passengers_option[0]
                # create state dict
                state = {
                    "optimal": optimal,
                    "map": game_map,
                    "taxis": taxis_option,
                    "passengers": passengers_option,
                }
                all_possible_states.append(state)
        self.all_possible_states = all_possible_states

    def get_all_possible_taxis_dicts(self, taxis_init: dict, game_map):
        """
        get all possible taxis dicts
        :param dict taxis_init: taxis dict in initial state of game.
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
        all_taxis_permutations = list(itertools.product(*taxis_options_lists))
        return all_taxis_permutations

    def get_all_legal_taxis_locations(self, game_map):
        """
        taxi location is any location on game_map that is not "I"
        """
        num_rows, num_cols = len(game_map), len(game_map[0])
        taxis_locations = [(i, j) for i in range(num_rows) for j in range(num_cols)]
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
        all_passengers_permutations = list(itertools.product(*passengers_options_lists))
        return all_passengers_permutations

    def get_all_legal_locations_by_passenger(self, passengers_init, taxis_init):
        """
        passenger location is on of the following:
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
                with each action and the probabilty to get to this next step.
        """
        for state in self.all_possible_states:
            legal_actions = self.actions(state)
            legal_actions.append("reset")
            legal_actions.append("terminate")
            self.all_possible_actions_for_state[state] = legal_actions
            for action in legal_actions:
                if action not in self.all_possible_actions:
                    self.all_possible_actions.append(action)
                next_states_with_probs = self.result_with_probs(state, action)
                for next_state, prob in next_states_with_probs.items():
                    self.Next_States_Probs[(state, action)] = (next_state, prob)

    def result_with_probs(self, state, action):
        """
        input: state, action
        output: next possible states and the probablilty for each one
        method:
            1.  Use the detrministic result function to get the next "regular" state.
            2.  Get Changes Probabilties per passnger
            3.  Get all possible new states with possible changes of destinations
                3a. Get new_state prob and possible destinations
                    where ths subset passengers may change their destination
                3b. Get all destinations permutations
                    (passengers have few possible goals to change to)
                3c. For each permutation get new state with updated destinations

        """
        possible_next_state_probs = {}
        # 1. Use the detrministic result function to get the next "regular" state.
        result_state = self.result(state, action)

        # 2. Get Changes Probabilties per passnger
        probs_by_pass = {}
        for pass_name, pass_details in result_state["passengers"].items():
            possible_goals = pass_details["possible_goals"]
            # prob_change_goal is to rechoice goal from possible goals uniformic.
            prob_rechoice_goal = pass_details["prob_change_goal"]
            prob_not = 1 - prob_rechoice_goal
            probs_by_pass[pass_name] = (prob_rechoice_goal, prob_not)

        # 3. Get all possible new states with possible changes of destinations
        all_passengers_subsets = powerset(state["passengers"].keys())
        all_passengers_subsets.append([])  # no passenger is rechoicing his destination
        # TODO: debug and make sure the loop doesnt skip the empty list
        for pass_subset_names in all_passengers_subsets:
            # 3a. Get new_state prob and possible destinations
            #     where ths subset passengers may change their destination
            new_state_prob = 1  # init prob
            possible_destinations_lists = []
            for pass_name, prob_rechoice_goal, prob_not in probs_by_pass.items():
                if pass_name in pass_subset_names:  # passenger rechoice destination
                    new_state_prob *= prob_rechoice_goal
                    possible_goals = list(result_state["passengers"]["possible_goals"])
                else:
                    new_state_prob *= prob_not
                    possible_goals = result_state["passengers"]["destination"]
                possible_destinations_lists.append(possible_goals)
            # 3b. Get all destinations permutations
            all_destinations_permu = list(
                itertools.product(*possible_destinations_lists)
            )
            # 3c. For each permutation get new state with updated destinations
            for destinations_permutation in all_destinations_permu:
                new_state = result_state.copy()
                for i, pass_name, pass_details in enumerate(
                    new_state["passengers"].items()
                ):
                    pass_details["destination"] = destinations_permutation[i]
                if new_state in possible_next_state_probs.keys():  # sum probs
                    existing_prob = possible_next_state_probs[new_state]
                    additional_prob = new_state_prob
                    possible_next_state_probs[new_state] = (
                        existing_prob + additional_prob
                    )
                else:
                    possible_next_state_probs[new_state] = new_state
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
        for t in range(self.max_turns_to_go):
            if t == 0:
                # End of game - no action is possible --> value is 0 for all states.
                for state in self.all_possible_states:
                    self.Values[t][state] = 0
            else:
                for state in self.all_possible_states:
                    # Update - Actions_Values dict
                    for action in self.all_possible_actions_for_state[state]:
                        # V(t, s) = max_over_a{
                        #              sum_over_next_s[prob(next_s, a)* V(t-1, next_s)] + Reward(a)}
                        value_of_action = self.Rewards[action]  # init
                        for next_state, prob in self.Next_States_Probs[(state, action)]:
                            value_of_action += prob * self.Values[t - 1][next_state]
                        self.Actions_Values[t][state][action] = value_of_action
                    # Find - best action and best value
                    possible_actions = self.Actions_Values[t][state].keys()
                    possible_values = self.Actions_Values[t][state].values()
                    best_value = np.max(possible_values)
                    best_action = possible_actions[np.argmax(possible_values)]
                    # Update  - Values and Best_Actions dicts
                    self.Values[t][state] = best_value
                    self.Best_Actions[t][state] = best_action

    def set_rewards_for_states(self):
        """
        For each action set the reward.
        """
        for action in self.all_possible_actions:
            if action == "reset":
                reward = -RESET_PENALTY
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
                map_size_height = state["map_size_height"]
                map_size_width = state["map_size_width"]
                map_matrix = state["map"]

                possible_locations = possible_locations_by_taxi[taxi_name]
                for new_location in possible_locations:
                    x, y = new_location
                    # 2. check that the taxi doesn't get out of the map
                    # 3. check that the taxi is on a passable tile
                    if (0 <= x < map_size_height) and (0 <= y < map_size_width):
                        if map_matrix[x][y] != "I":
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

    def actions(self, state: dict) -> Tuple[Tuple[Tuple]]:
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
        all_wait_action = tuple(
            [("wait", taxi_name) for taxi_name in state["taxis"].keys()]
        )
        assert all_wait_action in actions
        actions.remove(all_wait_action)

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

    def result(self, state: dict, action: Tuple[Tuple]) -> dict:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        result_state = deepcopy(state)

        for action_tuple in action:
            result_state = self._execute_action_tuple(result_state, action_tuple)

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
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplementedError


def dict_to_str(d: dict) -> str:
    d_str = str(d)
    return d_str


def str_to_dict(s: str) -> dict:
    j_dict = ast.literal_eval(s)
    return j_dict


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
    agent = OptimalTaxiAgent(initial=state)
    # agent.actions({"bla": "bla"})
    # agent.set_all_possible_states()
