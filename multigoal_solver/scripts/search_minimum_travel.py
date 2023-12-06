#!/usr/bin/env python3
import os
import pandas as pd
import random
import rospy
import rospkg
import yaml
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
import networkx as nx

class TravelOptimizer:
    def __init__(self):
        rospy.init_node('min_travel')

        self.service = rospy.Service('/optimize_path', Trigger, self.optimze_path)
        self.nodes = []

    # make dataframe symmetrical (if there is a connetion nodeX/ikY->nodeZ/ikW, then add  connetion nodeZ/ikW->nodeX/ikY
    def symmetric_entries(self,cost_db: pd.DataFrame):

        new_cost_db = cost_db.copy()
        for index, row in cost_db.iterrows():
            tmp = cost_db.loc[(cost_db['root'] == row['goal']) & (cost_db['root_ik_number']==row['goal_ik_number']) &
                              (cost_db['goal'] == row['root']) & (cost_db['goal_ik_number'] == row['root_ik_number'])]
            if (tmp.empty):
                new_row={'root': row['goal'],
                         'root_ik_number': row['goal_ik_number'],
                         'goal': row['root'],
                         'goal_ik_number': row['root_ik_number'],
                         'cost': row['cost']}
                new_cost_db = new_cost_db.append(new_row, ignore_index = True)
        return new_cost_db

    def is_fully_connected(self, cost_db):
        self.nodes = list(cost_db.root.unique())
        nodes_and_edges = {}
        for n in self.nodes:

            from_node_db = cost_db.loc[cost_db['root'] == n]
            attached_nodes = list(from_node_db.goal.unique())
            rospy.logdebug(f"from {n} you can go to {attached_nodes}")
            nodes_and_edges[n] = attached_nodes
        G = nx.Graph(nodes_and_edges)
        if nx.is_connected(G):
            rospy.logdebug("subarea are connected")
            return True
        else:
            rospy.logdebug(f"subarea are not connected. Sets {list(nx.connected_components(G))}")
            self.nodes = list(max(list(nx.connected_components(G))))
            self.nodes.sort()
            return False
        
    def optimze_path(self, req: TriggerRequest):

        resp = TriggerResponse()
        aware_shared_database = os.environ['AWARE_CNR_DB']

        blade_info = rospy.get_param("/blade_info")
        tool_name = rospy.get_param("/tool_name")
        max_refine_time = rospy.get_param("/travel_refine_time",60.0)
        file_name = os.path.join(aware_shared_database, "config", "results", blade_info['cloud_filename'])
        number_of_poi = rospy.get_param("/goals/number_of_poi")

        if number_of_poi>1:
            pingInfoFilePath = os.path.join(file_name+'_'+tool_name+'_costmap.ftr')
            cost_db = pd.read_feather(pingInfoFilePath, columns=None, use_threads=True)
            cost_db = self.symmetric_entries(cost_db)

            if not self.is_fully_connected(cost_db):
                resp.message = f'some subareas are not connected, using only {self.nodes}'
                rospy.logwarn(resp.message)

            ik_number = {}
            for n in self.nodes:
                ik_number[n] = int(1 + cost_db.loc[cost_db['root'] == n]['root_ik_number'].max())

            best_cost = float('inf')
            best_sequence = []

            t0 = rospy.Time().now()

            for idx in range(100):
                # for the first iteration try sorted node sequence, then try random node sequence
                travel_cost, travel_sequence = self.genClosestFirstTravel(self.nodes, ik_number, cost_db, best_cost,
                                                                          best_sequence, idx > 10)

                if travel_cost < best_cost:

                    t0 = rospy.Time().now()
                    best_cost = travel_cost
                    best_sequence = travel_sequence
                    print('- improve cost to', best_cost)
                    break

                if best_cost < float('inf') and (rospy.Time().now().to_sec()-t0.to_sec()) > max_refine_time:
                    break

            column = cost_db['cost']
            min_cost = cost_db['cost'].min()

            count = column[column > best_cost - (min_cost * (len(self.nodes) - 2))].count()
            #rospy.loginfo('row with too much cost', count, 'over', cost_db.shape[0])
            filter_db = cost_db.loc[cost_db['cost'] < (best_cost - (min_cost * (len(self.nodes) - 2)))]

            for idx in range(10):
                travel_cost, travel_sequence = self.genClosestFirstTravel(random_nodes=self.nodes,
                                                                          ik_number=ik_number,
                                                                          cost_db=filter_db,
                                                                          best_cost=best_cost,
                                                                          best_sequence=best_sequence)
                if travel_cost < best_cost:
                    best_cost = travel_cost
                    best_sequence = travel_sequence
                    print('- improve cost to', best_cost)
                    break
                if best_cost < float('inf') and (rospy.Time().now().to_sec()-t0.to_sec()) > max_refine_time:
                    break

            travel = list(best_sequence)
        else:
            node_name = rospy.get_param("/goals/node_prefix_name")+"0"
            travel=[{'node': node_name, 'ik': 0}]
        # remove typewriter effect
        rospy.logerr(travel)
        reverse_connection=0
        for it in range(0, len(travel) - 1):
            if travel[it]['node'] > travel[it + 1]['node']:
                reverse_connection += 1

        if (reverse_connection>=len(travel)*0.5):
            travel.reverse()

        rospy.logerr(travel)
        if len(travel):
            for idx in range(1, len(travel)):
                print(f'- {travel[idx - 1]["node"]}/iksol{travel[idx - 1]["ik"]} ==> '
                    f'{travel[idx]["node"]}/iksol{travel[idx]["ik"]}')

            rospy.set_param("/precompute_trees/travel", travel)

            blade_info = rospy.get_param("/blade_info")
            
            with open(os.path.join(file_name+'_'+tool_name+'_travel.yaml'), 'w') as file:
                documents = yaml.dump(travel, file)

            resp.success = True
            resp.message = f'path saved at: '+file_name+'_'+tool_name+'_travel.yaml'
        else:
            resp.success = False
            resp.message = f'It was impossible to optimize the inspection ({file_name},{tool_name})'

        return resp

    def genClosestFirstTravel(self, random_nodes, ik_number, cost_db, best_cost=float('inf'), best_sequence=[], shuffle = True):

        if shuffle:  # shuffle solution
            random.shuffle(random_nodes)
        else:  # try with alphabetical order
            random_nodes = sorted(random_nodes)

        # explored_nodes = [[]]
        for start in random_nodes:
            random_ik = [i for i in range(ik_number[start])]
            random.shuffle(random_ik)
            # explored_nodes.append(start)

            for start_ik in random_ik:
                remaining_nodes = self.nodes.copy()
                remaining_nodes.remove(start)

                sequence = [{'node': start, 'ik': float(start_ik)} for _ in range(len(remaining_nodes) + 1)]
                total_cost = 0

                remain_db = cost_db.loc[cost_db['goal'] != start]
                for i in range(len(remaining_nodes)):
                # while len(remaining_nodes) > 0:
                    db = remain_db.loc[(cost_db['root'] == start) & (remain_db['root_ik_number'] == start_ik)]

                    if db.shape[0] == 0:
                        total_cost = float('inf')
                        break

                    row = db.loc[db['cost'] == db['cost'].min()]
                    next = row.iloc[0]['goal']
                    next_ik = row.iloc[0]['goal_ik_number']
                    cost = row.iloc[0]['cost']
                    # remaining_nodes.remove(next)

                    total_cost += cost

                    if total_cost > best_cost:  # abort this solution because is worst than the best one
                        break
                    sequence[i+1] = {'node': next, 'ik': float(next_ik)}

                    start = next
                    start_ik = next_ik
                    remain_db = remain_db.loc[(remain_db['goal'] != start)]

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_sequence = sequence
                    print('improve cost to', best_cost)

        return best_cost, best_sequence


    # def genTravel(self, remaining_nodes, ik_number, cost_db, best_cost):
    #     possible_nodes = list(cost_db.goal.unique())
    #     start = random.choice(possible_nodes)
    #     remaining_nodes.remove(start)
    #
    #     start_ik = random.randrange(ik_number[start])
    #
    #     sequence = [{'node': start, 'ik': float(start_ik)}]
    #     total_cost = 0
    #     remain_db = cost_db.loc[(cost_db['goal'] != start)]
    #     while len(remaining_nodes) > 0:
    #         print('db size', remain_db.shape)
    #         db = remain_db.loc[(remain_db['root'] == start) & \
    #                            (remain_db['root_ik_number'] == start_ik)]
    #
    #         possible_nodes = list(db.goal.unique())
    #         if len(possible_nodes) == 0:
    #             return float('inf'), sequence
    #         next = random.choice(possible_nodes)
    #         db = db.loc[(db['goal'] == next)]
    #
    #         remaining_nodes.remove(next)
    #         while True:
    #             next_ik = random.randrange(ik_number[start])
    #             element = db.loc[(db['goal_ik_number'] == next_ik)]
    #             if len(element) > 0:
    #                 break
    #
    #         cost = element.iloc[0]['cost']
    #         total_cost += cost
    #         if total_cost > best_cost:  # abort this solution because is worst than the best one
    #             return float('inf'), sequence
    #         sequence.append({'node': next, 'ik': float(next_ik)})
    #         start = next
    #         start_ik = next_ik
    #         remain_db = remain_db.loc[(remain_db['goal'] != start)]
    #
    #     return total_cost, sequence


def main():
    node = TravelOptimizer()
    rospy.spin()

if __name__ == "__main__":
    main()
