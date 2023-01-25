#! /usr/bin/env python3

# =============================================================================
# Ant Colony System for Generalized Traveling Salesman Problem
# =============================================================================

import pandas as pd
import random
# import feather


def ant_colony_optimization(cost_db, nodes, ik_number, optimization_parameters):
    param = optimization_parameters.copy()
    best_ant = {
        "node": "",
        "ik": 0,
        "trail": [],
        "cost": float("inf"),
        "missing": [],
    }
    no_improv = 0
    obtained_at = 0
    for it in range(param["number_of_iterations"]):
        # Ants setup
        ants = [{"node": random.choice(nodes),
                 "ik": 0,
                 "trail": [],
                 "cost": 0,
                 "missing": nodes.copy()
                 }
                for idx in range(param["number_of_ants"])]
        
        for ant in ants:
            ant["ik"] = random.randint(0, ik_number[ant["node"]] - 1)
            ant["trail"].append({"node": ant["node"], "ik": ant["ik"]})
            ant["missing"].remove(ant["node"])

        # Simulation
        for tick in range(len(nodes) - 1):
            goto_next_node(cost_db, ants, beta=param["beta"], q0=param["q0"])
            local_update_ph(cost_db, ants, rho=param["rho"], ph_variation=param["ph_variation"])

        update_pweights(cost_db, beta=param["beta"])
        candidate_best_ant = get_best_ant(cost_db, ants)
        if best_ant["cost"] > candidate_best_ant["cost"]:
            best_ant = candidate_best_ant.copy()
            obtained_at = it + 1
            no_improv = 0
        else:
            no_improv += 1

        global_update_ph(cost_db, best_ant, rho=param["rho"])            
        check_ph_bounds(cost_db, ph_bounds=param["ph_bounds"])
        
        # Various print
        print("====================")
        print("iteration: {:4d}, Best ant obtained at {:3d}".format(it + 1, obtained_at))
        print_ant(best_ant)
        print("Other costs: ")
        cc = 0
        for ant in ants:
            print("[{:8s}, ik {:3d}: {:10.4f}] ".format(ant["trail"][0]["node"], ant["trail"][0]["ik"], ant["cost"]), end="")
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

    if it >= param["number_of_iterations"]-1:
        exit_reason = "Max number of iteration reached ({})".format(param["number_of_iterations"])
    return (best_ant, obtained_at, exit_reason)


def get_best_ant(cost_db, ants):
    best_cost = float("inf")
    best_ant = ants[0]
    for a in ants:
        if a["cost"] < best_cost and len(a["missing"]) == 0:
            best_ant = a
            best_cost = a["cost"]

    return best_ant


def normalization_term_probability(cost_db, ant, beta):
    cost_inverse_db = cost_db.loc[(cost_db["root"] == ant["node"])
                                  & (cost_db["root_ik_number"] == ant["ik"])
                                  & (cost_db["goal"].isin(ant["missing"])),
                                  "cost"].rdiv(1).pow(beta)

    return float(cost_db.loc[(cost_db["root"] == ant["node"])
                             & (cost_db["root_ik_number"] == ant["ik"])
                             & (cost_db["goal"].isin(ant["missing"])),
                             "ph"]
                 .mul(cost_inverse_db)
                 .sum()
                 )


def update_pweights(cost_db, beta):
    cost_db["pweight"] = cost_db["cost"].rdiv(1).pow(beta).mul(cost_db["ph"])

def check_ph_bounds(cost_db, ph_bounds):
    cost_db.loc[cost_db["ph"] < ph_bounds[0], "ph"] = ph_bounds[0]
    cost_db.loc[cost_db["ph"] > ph_bounds[1], "ph"] = ph_bounds[1]

def local_update_ph(cost_db, ants, rho, ph_variation):
    for ant in ants:
        cost_db.loc[(cost_db["root"] == ant["trail"][-2]["node"])
                    & (cost_db["root_ik_number"] == ant["trail"][-2]["ik"])
                    & (cost_db["goal"] == ant["trail"][-1]["node"])
                    & (cost_db["goal_ik_number"] == ant["trail"][-1]["ik"]),
                    "ph"] = float(
                                  cost_db.loc[(cost_db["root"] == ant["trail"][-2]["node"])
                                  & (cost_db["root_ik_number"] == ant["trail"][-2]["ik"])
                                  & (cost_db["goal"] == ant["trail"][-1]["node"])
                                  & (cost_db["goal_ik_number"] == ant["trail"][-1]["ik"]),
                                  "ph"]
                                  ) * (1 - rho) + rho * ph_variation

def global_update_ph(cost_db, best_ant, rho):
    cost_db["ph"] *= (1 - rho)
    trail_idx = get_trail_idx(cost_db, best_ant["trail"])
    cost_db.loc[trail_idx, "ph"] += rho / get_trail_cost(cost_db, best_ant)


def possible_next_node_idx(cost_db, ant):
     return list(cost_db.loc[(cost_db["root"] == ant["node"])
                             & (cost_db["root_ik_number"] == ant["ik"])
                             & (cost_db["goal"].isin(ant["missing"]))
                             ].index)
    

def next_node_probsum(cost_db, select_idx, ant, beta):
    # probability = (cost * visibility ^ beta) / sum_probablities
    # visibility = 1/pheromone
    return list(cost_db.loc[select_idx, "pweight"]
                .mul(
                    cost_db.loc[select_idx, "cost"]
                    .rdiv(1)
                    .pow(beta)
                )
                .div(
                    normalization_term_probability(
                        cost_db.loc[select_idx],
                        ant,
                        beta=beta
                    )
                )
                .cumsum()
                )


def goto_next_node(cost_db, ants, beta, q0):
    for ant in ants:
        select_idx = possible_next_node_idx(cost_db, ant)       
        cumsum_list = next_node_probsum(cost_db, select_idx, ant, beta)
        if random.random() > q0:
            idx = random.choices(select_idx, cum_weights=cumsum_list, k=1)[0]
        else:
            idx = cost_db.loc[select_idx, "pweight"].idxmax()

        ant["node"] = str(cost_db.at[idx, "goal"])
        ant["ik"] = int(cost_db.at[idx, "goal_ik_number"])
        ant["trail"].append({"node": ant["node"], "ik": ant["ik"]})
        update_cost(cost_db, ant)
        ant["missing"].remove(ant["node"])


def update_cost(cost_db, ant):
    if len(ant["trail"]) > 1:
        ant["cost"] += float(
            cost_db.loc[(cost_db["root"] == ant["trail"][-2]["node"])
                        & (cost_db["root_ik_number"] == ant["trail"][-2]["ik"])
                        & (cost_db["goal"] == ant["trail"][-1]["node"])
                        & (cost_db["goal_ik_number"] == ant["trail"][-1]["ik"]),
                        "cost"])


def get_trail_cost(cost_db, ant):
    total_cost = 0
    vm1 = ant["trail"][0]
    for v in ant["trail"][1:]:
        total_cost = float(cost_db.loc[
                           (cost_db["root"] == vm1["node"])
                           & (cost_db["root_ik_number"] == vm1["ik"])
                           & (cost_db["goal"] == v["node"])
                           & (cost_db["goal_ik_number"] == v["ik"]),
                           "cost"]
                           ) + total_cost
        vm1 = v

    return total_cost


def get_trail_idx(cost_db, tour):
    trail_idx = []
    vm1 = tour[0]
    for v in tour[1:]:
        trail_idx.append(list(cost_db.loc[(cost_db["root"] == vm1["node"])
                                      & (cost_db["root_ik_number"] == vm1["ik"])
                                      & (cost_db["goal"] == v["node"])
                                      & (cost_db["goal_ik_number"] == v["ik"])]
                              .index
                              )[0]
                         )
        vm1 = v
    return trail_idx


def print_ant(ant):
    print("[Ant] Cost: {}".format(ant["cost"]))
    print("({:8s},{:2d})".format(ant["trail"][0]["node"], ant["trail"][0]["ik"]), end="")
    cc = 0
    for t in ant["trail"][1:]:
        print(" --> ({:8s},{:2d})".format(t["node"], t["ik"]), end="")
        if cc % 4 == 3:
            cc += 1
            print("")
    print("")


def main():
    pingInfoFilePath = "./costmap.ftr"
    cost_db = pd.read_feather(pingInfoFilePath, columns=None, use_threads=True)

    # Number of connections:
    # (n_cluster - 1) * n_nodes_per_cluster^2 * n_cluster
    
    nodes = list(cost_db.root.unique())
    ik_number = {}
    for n in nodes:
        ik_number[n] = int(1 + cost_db.loc[cost_db["root"] == n]["root_ik_number"].max())

    # Solver parameters
    solver_param = {"number_of_ants": 10,
                    "number_of_iterations": 100,
                    "beta": 5,
                    "rho": 0.5,
                    "q0": 0.5,
                    "ph_init": 1/(12*len(nodes)),
                    "ph_variation": 1/(12*len(nodes)),
                    "ph_bounds": tuple([x/(12*len(nodes)) for x in [0.1, 1.05]]),
                    "max_no_improv": 10, #float("inf"),
                    }

    # Add Pheromone column
    cost_db.insert(cost_db.shape[1], "ph", solver_param["ph_init"] * cost_db.shape[0])

    # Add Heuristic column
    cost_db.insert(cost_db.shape[1], "pweight", [1.0] * cost_db.shape[0])
    update_pweights(cost_db, beta=solver_param["beta"])

    best_ant, obtained_at, exit_reason = ant_colony_optimization(cost_db, nodes, ik_number, solver_param)

    print("Exit reason: {}".format(exit_reason))
    print_ant(best_ant)
        

if __name__ == "__main__":
    main()
