#! /usr/bin/env python3

# =============================================================================
# Ant Colony System Optimization for Generalized Traveling Salesman Problem
# =============================================================================

import pandas as pd
import random
from typing import List, Dict
import time

class Ant():
    def __init__(self,
                 cost_db: pd.DataFrame,
                 node: str,
                 ik: int,
                 missing: List[str] = [],
                 trail: List[Dict] = [],
                 cost: float = 0,
                 ) -> None:
        self.cost_db = cost_db
        self.node = node
        self.ik = ik
        self.trail = trail.copy()
        self.cost = cost
        self.missing = missing.copy()
        
    
    def setNode(self, node: str) -> None:
        self.node = node
    
    
    def setIk(self, ik: int) -> None:
        self.ik = ik


    def addNodeToTrail(self) -> None:
        self.trail.append({"node": self.node, "ik": self.ik})


    def updateCost(self) -> None:
        if len(self.trail) > 1:
            self.cost += float(
                self.cost_db.loc[(self.cost_db["root"] == self.trail[-2]["node"])
                                 & (self.cost_db["root_ik_number"] == self.trail[-2]["ik"])
                                 & (self.cost_db["goal"] == self.trail[-1]["node"])
                                 & (self.cost_db["goal_ik_number"] == self.trail[-1]["ik"]),
                                 "cost"])
            
    def updateMissing(self) -> None:
        self.missing.remove(self.node)
        
    
    def getTrailCost(self):
        total_cost = 0
        vm1 = self.trail[0]
        for v in self.trail[1:]:
            total_cost = float(self.cost_db.loc[
                               (self.cost_db["root"] == vm1["node"])
                               & (self.cost_db["root_ik_number"] == vm1["ik"])
                               & (self.cost_db["goal"] == v["node"])
                               & (self.cost_db["goal_ik_number"] == v["ik"]),
                               "cost"]
                               ) + total_cost
            vm1 = v
        return total_cost
    
    
    def getTrailIndex(self):
        trail_idx = []
        vm1 = self.trail[0]
        for v in self.trail[1:]:
            trail_idx.append(list(self.cost_db.loc[(self.cost_db["root"] == vm1["node"])
                                                   & (self.cost_db["root_ik_number"] == vm1["ik"])
                                                   & (self.cost_db["goal"] == v["node"])
                                                   & (self.cost_db["goal_ik_number"] == v["ik"])]
                                  .index
                                  )[0]
                             )
            vm1 = v
        return trail_idx


    def printAnt(self):
        print("[Ant] Cost: {}".format(self.cost))
        print("({:8s},{:2d})".format(self.trail[0]["node"], self.trail[0]["ik"]), end="")
        cc = 0
        for t in self.trail[1:]:
            print(" --> ({:8s},{:2d})".format(t["node"], t["ik"]), end="")
            if cc % 4 == 3:
                cc += 1
                print("")
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
                 ph_bounds: (float, float),
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
        
        self.nodes = list(cost_db.root.unique())
        self.ik_number = {}
        for n in self.nodes:
            self.ik_number[n] = int(1 + cost_db.loc[self.cost_db["root"] == n]["root_ik_number"].max())
        
        # Add Pheromone column
        self.cost_db.insert(cost_db.shape[1], "ph", self.ph_init * self.cost_db.shape[0])

        # Add Heuristic column
        self.cost_db.insert(cost_db.shape[1], "pweight", [1.0] * self.cost_db.shape[0])
        self.updatePWeights()
        
        self.best_ant = Ant(self.cost_db, node="", ik=0, cost=float("inf"))


    def run(self) -> (float, int, str):
        no_improv = 0
        obtained_at = 0
        for it in range(self.number_of_iterations):
            self.ants = [Ant(self.cost_db,
                             node=random.choice(self.nodes),
                             ik=0,
                             missing=self.nodes.copy()
                             ) for idx in range(self.number_of_ants)
                         ]
            
            for ant in self.ants:
                ant.setIk(random.randint(0, self.ik_number[ant.node] - 1))
                ant.addNodeToTrail()
                ant.updateMissing()

            # Simulation
            for tick in range(len(self.nodes) - 1):
                self.gotoNextNode()
                self.localUpdatePh()

            self.updatePWeights()
            candidate_best_ant = self.getBestAnt()
            if self.best_ant.cost > candidate_best_ant.cost:
                self.best_ant = candidate_best_ant # Copy ??
                obtained_at = it + 1
                no_improv = 0
            else:
                no_improv += 1

            self.globalUpdatePh(self.best_ant)
            self.checkPhBounds()
            
            # Various print
            print("====================")
            print("iteration: {:4d}, Best ant obtained at {:3d}".format(it + 1, obtained_at))
            self.best_ant.printAnt()
            print("Other costs: ")
            cc = 0
            for ant in self.ants:
                print("[{:8s}, ik {:3d}: {:10.4f}] ".format(ant.trail[0]["node"], ant.trail[0]["ik"], ant.cost), end="")
                if cc % 2 == 1:
                    print(" ")
                cc += 1
            print(" ")


            # # DEBUG
            # if it >= 20:
            #     pass

            # if no_improv > param["max_no_improv"]:
            #     exit_reason = "No improvement after {} iterations".format(param["max_no_improv"])
            #     break

        if it >= self.number_of_iterations-1:
            exit_reason = "Max number of iteration reached ({})".format(self.number_of_iterations)
        return (self.best_ant, obtained_at, exit_reason)
        
    
    def updatePWeights(self) -> None:
        self.cost_db["pweight"] = self.cost_db["cost"].rdiv(1).pow(self.beta).mul(self.cost_db["ph"])


    def getBestAnt(self) -> Ant:
        best_cost = float("inf")
        best_ant = self.ants[0]
        for a in self.ants:
            if a.cost < best_cost and len(a.missing) == 0:
                best_ant = a
                best_cost = a.cost

        return best_ant


    def getNeighbourPWeightSum(self, ant: Ant, cost_db: pd.DataFrame=None) -> float:
        if cost_db is None:
            cost_db = self.cost_db

        cost_inverse_db = cost_db.loc[(cost_db["root"] == ant.node)
                                      & (cost_db["root_ik_number"] == ant.ik)
                                      & (cost_db["goal"].isin(ant.missing)),
                                      "cost"].rdiv(1).pow(self.beta)

        return float(cost_db.loc[(cost_db["root"] == ant.node)
                                 & (cost_db["root_ik_number"] == ant.ik)
                                 & (cost_db["goal"].isin(ant.missing)),
                                 "ph"]
                     .mul(cost_inverse_db)
                     .sum()
                     )


    def checkPhBounds(self) -> None:
        self.cost_db.loc[self.cost_db["ph"] < self.ph_bounds[0], "ph"] = self.ph_bounds[0]
        self.cost_db.loc[self.cost_db["ph"] > self.ph_bounds[1], "ph"] = self.ph_bounds[1]


    def localUpdatePh(self) -> None:
        for ant in self.ants:
            self.cost_db.loc[(self.cost_db["root"] == ant.trail[-2]["node"])
                             & (self.cost_db["root_ik_number"] == ant.trail[-2]["ik"])
                             & (self.cost_db["goal"] == ant.trail[-1]["node"])
                             & (self.cost_db["goal_ik_number"] == ant.trail[-1]["ik"]),
                             "ph"] = float(
                                      self.cost_db.loc[(self.cost_db["root"] == ant.trail[-2]["node"])
                                                       & (self.cost_db["root_ik_number"] == ant.trail[-2]["ik"])
                                                       & (self.cost_db["goal"] == ant.trail[-1]["node"])
                                                       & (self.cost_db["goal_ik_number"] == ant.trail[-1]["ik"]),
                                                       "ph"]
                                           ) * (1 - self.rho) + self.rho * self.ph_variation

                                 
    def globalUpdatePh(self, best_ant: Ant = None) -> None:
        if best_ant is None:
            best_ant = self.getBestAnt()
        self.cost_db["ph"] *= (1 - self.rho)
        trail_idx = best_ant.getTrailIndex()
        self.cost_db.loc[trail_idx, "ph"] += self.rho / best_ant.getTrailCost()


    def getPossibleNextNodeIndex(self, ant: Ant) -> List[int]:
         return list(self.cost_db.loc[(self.cost_db["root"] == ant.node)
                                      & (self.cost_db["root_ik_number"] == ant.ik)
                                      & (self.cost_db["goal"].isin(ant.missing))
                                      ].index)


    def getNextNodeProbCumsum(self, select_idx: List[int], ant: Ant):
        # probability = (cost * visibility ^ beta) / sum_probablities
        # visibility = 1/pheromone
        return list(self.cost_db.loc[select_idx, "pweight"]
                    .mul(
                         self.cost_db.loc[select_idx, "cost"]
                         .rdiv(1)
                         .pow(self.beta)
                         )
                    .div(
                         self.getNeighbourPWeightSum(
                                                     ant,
                                                     self.cost_db.loc[select_idx]
                                                     )
                         )
                    .cumsum()
                    )


    def gotoNextNode(self):
        for ant in self.ants:
            select_idx = self.getPossibleNextNodeIndex(ant)
            cumsum_list = self.getNextNodeProbCumsum(select_idx, ant)
            if random.random() > self.q0:
                idx = random.choices(select_idx, cum_weights=cumsum_list, k=1)[0]
            else:
                idx = self.cost_db.loc[select_idx, "pweight"].idxmax()

            ant.setNode(str(self.cost_db.at[idx, "goal"]))
            ant.setIk(int(self.cost_db.at[idx, "goal_ik_number"]))
            ant.addNodeToTrail()
            ant.updateCost()
            ant.updateMissing()


def main():
    pingInfoFilePath = "./costmap2.ftr"
    cost_db = pd.read_feather(pingInfoFilePath, columns=None, use_threads=True)

    # Number of connections:
    # (n_cluster - 1) * n_nodes_per_cluster^2 * n_cluster
    
    nodes = list(cost_db.root.unique())
    acs = AntColonySystem(cost_db,
                          number_of_ants=10,
                          number_of_iterations=100,
                          beta=5,
                          rho=0.5,
                          q0=0.5,
                          ph_init=1/(12*len(nodes)),
                          ph_variation=1/(12*len(nodes)),
                          ph_bounds=tuple([x/(12*len(nodes)) for x in [0.1, 1.05]]),
                          max_no_improv=float("inf"),
                          )
    tic = time.perf_counter()
    best_ant, obtained_at, exit_reason = acs.run()
    toc = time.perf_counter()
    print("Timing: {}".format(toc-tic))
    print("Exit reason: {}".format(exit_reason))
    best_ant.printAnt()


def test10():
    pingInfoFilePath = "./costmap.ftr"
    cost_db = pd.read_feather(pingInfoFilePath, columns=None, use_threads=True)
    nodes = list(cost_db.root.unique())
    acs = AntColonySystem(cost_db,
                          number_of_ants=10,
                          number_of_iterations=100,
                          beta=5,
                          rho=0.5,
                          q0=0.5,
                          ph_init=1/(12*len(nodes)),
                          ph_variation=1/(12*len(nodes)),
                          ph_bounds=tuple([x/(12*len(nodes)) for x in [0.1, 1.05]]),
                          max_no_improv=float("inf"),
                          )
    best_list = []
    for idx in range(10):
        best_ant, obtained_at, exit_reason = acs.run()
        best_list.append(best_ant)
        print("Exit reason: {}".format(exit_reason))
        #best_ant.printAnt()
        
    for ba in best_list:
        ba.printAnt()


if __name__ == "__main__":
    main()
