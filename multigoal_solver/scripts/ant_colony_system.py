#! /usr/bin/env python3

# =============================================================================
# Ant Colony System Optimization for Generalized Traveling Salesman Problem
# =============================================================================

import rospy
import random
from typing import List, Dict, Tuple

import pandas as pd


class Ant():
    def __init__(self,
                 cost_db: pd.DataFrame,
                 node: str,
                 ik: int,
                 missing: List[str] = None,
                 trail: List[Dict] = None,
                 cost: float = 0,
                 ) -> None:
        self.cost_db = cost_db
        self.node = node
        self.ik = ik

        if trail is None:
            self.trail = []
        else:
            self.trail = trail.copy()

        self.cost = cost

        if missing is None:
            self.missing = []
        else:
            self.missing = missing.copy()

    def set_node(self, node: str) -> None:
        self.node = node
        
    def set_ik(self, ik: int) -> None:
        self.ik = ik

    def move(self, node: str, ik: str) -> None:
        self.node = node
        self.ik = ik
        self.add_node_to_trail()
        self.update_cost()
        self.update_missing()

    def add_node_to_trail(self) -> None:
        self.trail.append({"node": self.node, "ik": self.ik})

    def update_cost(self) -> None:
        if len(self.trail) > 1:
            self.cost += float(
                self.cost_db.loc[
                    (self.cost_db["root"] == self.trail[-2]["node"])
                    & (self.cost_db["root_ik_number"] == self.trail[-2]["ik"])
                    & (self.cost_db["goal"] == self.trail[-1]["node"])
                    & (self.cost_db["goal_ik_number"] == self.trail[-1]["ik"]),
                    "cost"
                ].iloc[0]
            )

    def update_missing(self) -> None:
        self.missing.remove(self.node)

    def get_trail_cost(self) -> float:
        total_cost = 0
        vm1 = self.trail[0]
        for v in self.trail[1:]:
            total_cost = float(
                self.cost_db.loc[
                    (self.cost_db["root"] == vm1["node"])
                    & (self.cost_db["root_ik_number"] == vm1["ik"])
                    & (self.cost_db["goal"] == v["node"])
                    & (self.cost_db["goal_ik_number"] == v["ik"]),
                    "cost"
                ].iloc[0]
            ) + total_cost
            vm1 = v
        return total_cost

    def get_trail_index(self) -> List[int]:
        trail_idx = []
        vm1 = self.trail[0]
        for v in self.trail[1:]:
            trail_idx.append(
                list(
                    self.cost_db.loc[(self.cost_db["root"] == vm1["node"])
                                     & (self.cost_db["root_ik_number"] == vm1["ik"])
                                     & (self.cost_db["goal"] == v["node"])
                                     & (self.cost_db["goal_ik_number"] == v["ik"])
                                     ].index)[0]
            )
            vm1 = v
        return trail_idx

    def print_ant(self) -> None:
        print("[Ant] Cost: {}".format(self.cost))
        print("[{:8s},{:2d}]".format(self.trail[0]["node"], self.trail[0]["ik"]))
        for t in self.trail[1:]:
            print("[{:8s}, {:2d}]".format(t["node"], t["ik"]))
        print("")


class AntColonySystem():
    def __init__(self,
                 cost_db: pd.DataFrame,
                 number_of_ants: int,
                 number_of_iterations: int,
                 beta: float,
                 rho: float,
                 q0: float,
                 ph_init: float,
                 ph_variation: float,
                 ph_bounds: Tuple[float, float],
                 max_no_improv: int) -> None:

        self.cost_db = cost_db
        self.number_of_ants = number_of_ants
        self.number_of_iterations = number_of_iterations
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.ph_init = ph_init
        self.ph_variation = ph_variation
        self.ph_bounds = ph_bounds
        self.max_no_improv = max_no_improv
        self.ants = []

        self.nodes = list(cost_db.root.unique())
        self.ik_number = {}
        for n in self.nodes:
            self.ik_number[n] = int(1 + cost_db.loc[self.cost_db["root"] == n]["root_ik_number"].max())

        # Add Pheromone column
        self.cost_db.insert(cost_db.shape[1], "ph", self.ph_init * self.cost_db.shape[0])

        # Add Heuristic column
        self.cost_db.insert(cost_db.shape[1], "pweight", [1.0] * self.cost_db.shape[0])
        self.update_pweights()

        self.best_ant = Ant(self.cost_db, node="", ik=0, cost=float("inf"))

    def run(self) -> Tuple[float, int, str]:
        no_improv = 0
        obtained_at = 0
        for it in range(self.number_of_iterations):
            self.ants = [self.create_random_ant() for idx in range(self.number_of_ants)]

            # Simulation
            lost_ants = []
            for _ in range(len(self.nodes) - 1):
                lost_ants = self.goto_next_node()
                for dying_ant in lost_ants:
                    self.ants.remove(dying_ant)
                if not self.ants:
                    break
                self.local_update_ph()

            self.update_pweights()
            
            if not self.ants:
                print("[{}] All ants died".format(it))
                no_improv += 1
                continue

            candidate_best_ant = self.get_best_ant()
            if self.best_ant.cost > candidate_best_ant.cost:
                self.best_ant = candidate_best_ant
                obtained_at = it + 1
                no_improv = 0
            else:
                no_improv += 1

            self.global_update_ph(self.best_ant)
            self.check_ph_bounds()

            # Print for debug purposes
            # self.print_iteration(it, obtained_at, self.ants)
            print("[{}] Best cost = {}".format(it, self.best_ant.cost))

        if it >= self.number_of_iterations - 1:
            exit_reason = "Max number of iteration reached ({})".format(self.number_of_iterations)
        return (self.best_ant, obtained_at, exit_reason)

    def create_random_ant(self) -> Ant:
        ant = Ant(
            self.cost_db,
            node=random.choice(self.nodes),
            ik=0,
            missing=self.nodes.copy()
        )
        ant.set_ik(random.randint(0, self.ik_number[ant.node] - 1))
        ant.add_node_to_trail()
        ant.update_missing()
        return ant

    def update_pweights(self) -> None:
        self.cost_db["pweight"] = self.cost_db["cost"].rdiv(1).pow(self.beta).mul(self.cost_db["ph"])

    def get_best_ant(self) -> Ant:
        best_cost = float("inf")
        best_ant = self.ants[0]
        for a in self.ants:
            if a.cost < best_cost and len(a.missing) == 0:
                best_ant = a
                best_cost = a.cost

        return best_ant

    def get_neighbour_pweight_sum(self, ant: Ant, cost_db: pd.DataFrame = None) -> float:
        if cost_db is None:
            cost_db = self.cost_db

        cost_inverse_db = cost_db.loc[
            (cost_db["root"] == ant.node)
            & (cost_db["root_ik_number"] == ant.ik)
            & (cost_db["goal"].isin(ant.missing)),
            "cost"
        ].rdiv(1).pow(self.beta)

        return float(
            cost_db.loc[(cost_db["root"] == ant.node)
                        & (cost_db["root_ik_number"] == ant.ik)
                        & (cost_db["goal"].isin(ant.missing)),
                        "ph"]
            .mul(cost_inverse_db)
            .sum()
        )

    def check_ph_bounds(self) -> None:
        self.cost_db.loc[self.cost_db["ph"] < self.ph_bounds[0], "ph"] = self.ph_bounds[0]
        self.cost_db.loc[self.cost_db["ph"] > self.ph_bounds[1], "ph"] = self.ph_bounds[1]

    def local_update_ph(self) -> None:
        for ant in self.ants:
            self.cost_db.loc[
                (self.cost_db["root"] == ant.trail[-2]["node"])
                & (self.cost_db["root_ik_number"] == ant.trail[-2]["ik"])
                & (self.cost_db["goal"] == ant.trail[-1]["node"])
                & (self.cost_db["goal_ik_number"] == ant.trail[-1]["ik"]),
                "ph"
            ] = float(
                self.cost_db.loc[
                    (self.cost_db["root"] == ant.trail[-2]["node"])
                    & (self.cost_db["root_ik_number"] == ant.trail[-2]["ik"])
                    & (self.cost_db["goal"] == ant.trail[-1]["node"])
                    & (self.cost_db["goal_ik_number"] == ant.trail[-1]["ik"]),
                    "ph"].iloc[0]
            ) * (1 - self.rho) + self.rho * self.ph_variation

    def global_update_ph(self, best_ant: Ant = None) -> None:
        if best_ant is None:
            best_ant = self.get_best_ant()
        self.cost_db["ph"] *= (1 - self.rho)
        trail_idx = best_ant.get_trail_index()
        self.cost_db.loc[trail_idx, "ph"] += self.rho / best_ant.get_trail_cost()

    def get_possible_next_node_index(self, ant: Ant) -> List[int]:
        return list(self.cost_db.loc[(self.cost_db["root"] == ant.node)
                                     & (self.cost_db["root_ik_number"] == ant.ik)
                                     & (self.cost_db["goal"].isin(ant.missing))
                                     ].index)

    def get_next_node_prob_cumsum(self, select_idx: List[int], ant: Ant) -> List[float]:
        # probability = (cost * visibility ^ beta) / sum_probablities
        # visibility = 1/pheromone
        return list(
            self.cost_db.loc[select_idx, "pweight"]
            .mul(
                self.cost_db.loc[select_idx, "cost"]
                .rdiv(1)
                .pow(self.beta)
            )
            .div(
                self.get_neighbour_pweight_sum(
                    ant,
                    self.cost_db.loc[select_idx]
                )
            )
            .cumsum()
        )

    def goto_next_node(self) -> None:
        lost_ants = []
        for ant in self.ants:
            select_idx = self.get_possible_next_node_index(ant)
            if not select_idx:
                lost_ants.append(ant)
                continue
            cumsum_list = self.get_next_node_prob_cumsum(select_idx, ant)
            if random.random() > self.q0:
                idx = random.choices(select_idx, cum_weights=cumsum_list, k=1)[0]
            else:
                idx = self.cost_db.loc[select_idx, "pweight"].idxmax()

            ant.move(str(self.cost_db.at[idx, "goal"]),
                     int(self.cost_db.at[idx, "goal_ik_number"]))

        return lost_ants


    def print_iteration(self,
                        it: int,
                        obtained_at: int) -> None:
        # Various print
        print("====================")
        print("iteration: {:4d}, Best ant obtained at {:3d}".format(it + 1, obtained_at))
        self.best_ant.print_ant()
        print("Other costs: ")
        cc = 0
        for ant in self.ants:
            print("[{:8s}, ik {:3d}: {:10.4f}] ".format(ant.trail[0]["node"],
                                                        ant.trail[0]["ik"],
                                                        ant.cost),
                  end="")
            if cc % 2 == 1:
                print(" ")
            cc += 1
        print(" ")

    def nearest_neighbour(self) -> Ant:
        best_cost = self.best_ant.cost
        start = self.best_ant.trail[0]["node"]
        random_ik = list(range(0, self.ik_number[start]))
        random.shuffle(random_ik)
        for start_ik in random_ik:
            remaining_nodes = self.nodes.copy()
            remaining_nodes.remove(start)

            sequence = [{'node': start, 'ik': start_ik}]
            total_cost = 0

            remain_db = self.cost_db.loc[(self.cost_db['goal'] != start)]

            while len(remaining_nodes) > 0:
                db = remain_db.loc[(self.cost_db['root'] == start)
                                   & (remain_db['root_ik_number'] == start_ik)]

                if (db.shape[0]==0):
                    total_cost=float('inf')
                    break

                row = db.loc[db['cost'] == db['cost'].min()]
                next_goal = row.iloc[0]['goal']
                next_ik = row.iloc[0]['goal_ik_number']
                cost = row.iloc[0]['cost']
                remaining_nodes.remove(next_goal)

                total_cost += cost
                if (total_cost > best_cost):
                    break
                sequence.append({'node': next_goal, 'ik': next_ik})

                start = next_goal
                start_ik = next_ik
                remain_db = remain_db.loc[(remain_db['goal'] != start)]

            if (total_cost < best_cost):
                best_cost = total_cost
                best_sequence = sequence

        return Ant(self.cost_db,
                   node=best_sequence[-1]["node"],
                   ik=best_sequence[-1]["ik"],
                   missing=[],
                   trail=best_sequence.copy(),
                   cost=best_cost)


def test_ACS(feather_db_path: str, number_of_runs: int = 1) -> None:
    """
    Test routine
    """
    cost_db = pd.read_feather(feather_db_path, columns=None, use_threads=True)
    nodes = list(cost_db.root.unique())
    acs = AntColonySystem(cost_db,
                          number_of_ants=10,
                          number_of_iterations=100,
                          beta=5,
                          rho=0.5,
                          q0=0.5,
                          ph_init=1 / (12 * len(nodes)),
                          ph_variation=1 / (12 * len(nodes)),
                          ph_bounds=tuple([x / (12 * len(nodes)) for x in [0.1, 1.05]]),
                          max_no_improv=10,
                          )
    best_list = []
    for _ in range(number_of_runs):
        best_ant, _, _ = acs.run()
        best_list.append(best_ant)
        # print("Exit reason: {}".format(exit_reason))

    for ba in best_list:
        ba.print_ant()

    #rospy.set_param("/precompute_trees/travel",travel)

def main() -> None:
    feather_db_path = "./costmap3.ftr"
    cost_db = pd.read_feather(feather_db_path, columns=None, use_threads=True)
    nodes = list(cost_db.root.unique())
    acs = AntColonySystem(cost_db,
                          number_of_ants=10,
                          number_of_iterations=100,
                          beta=5,
                          rho=0.5,
                          q0=0.5,
                          ph_init=1 / (12 * len(nodes)),
                          ph_variation=1 / (12 * len(nodes)),
                          ph_bounds=tuple([x / (12 * len(nodes)) for x in [0.1, 1.05]]),
                          max_no_improv=10,
                          )
    best_ant, _, _ = acs.run()
    best_ant.print_ant()
    rospy.set_param("/precompute_trees/travel", best_ant.trail)

if __name__ == "__main__":
    #pd.set_option("mode.chained_assignment","warn")
    main()
