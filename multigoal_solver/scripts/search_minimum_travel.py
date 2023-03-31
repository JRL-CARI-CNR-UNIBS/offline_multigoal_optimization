#!/usr/bin/env python3
import pandas as pd
import random
import rospy
import feather
import signal
import rospkg
import yaml
#sudo apt install python3-feather-format
#pip3 instal pyarrow

def genClosestFirstTravel(nodes,ik_number,cost_db,best_cost=float('inf'),best_sequence=[]):
    random_nodes=nodes.copy()
    random.shuffle(random_nodes)

    explored_nodes=list([])
    for start in random_nodes:
        random_ik=list(range(0,ik_number[start]))
        random.shuffle(random_ik)
        explored_nodes.append(start)

        for start_ik in random_ik:
            remaining_nodes=nodes.copy()
            remaining_nodes.remove(start)

            sequence=[{'node': start,'ik': float(start_ik)}]
            total_cost=0

            remain_db=cost_db.loc[(cost_db['goal'] != start)]
            while len(remaining_nodes)>0:
                db=remain_db.loc[(cost_db['root'] == start) &\
                (remain_db['root_ik_number'] == start_ik)]

                if (db.shape[0]==0):
                    total_cost=float('inf')
                    break

                row=db.loc[db['cost']==db['cost'].min()]
                next = row.iloc[0]['goal']
                next_ik = row.iloc[0]['goal_ik_number']
                cost = row.iloc[0]['cost']
                remaining_nodes.remove(next)

                total_cost+=cost

                if (total_cost>best_cost):  # abort this solution because is worst than the best one
                    break
                sequence.append({'node': next,'ik': float(next_ik)})

                start=next
                start_ik=next_ik
                remain_db=remain_db.loc[(remain_db['goal'] != start)]

            if (total_cost<best_cost):
                best_cost=total_cost
                best_sequence=sequence
                print('improve cost to',best_cost)
    return best_cost, best_sequence


def genTravel(nodes,ik_number,cost_db,best_cost):
    remaining_nodes=nodes.copy()
    possible_nodes=list(cost_db.goal.unique())
    start = random.choice(possible_nodes)
    remaining_nodes.remove(start)

    start_ik = random.randrange(ik_number[start])

    sequence=[{'node': start,'ik': float(start_ik)}]
    total_cost=0
    remain_db=cost_db.loc[(cost_db['goal'] != start)]
    while len(remaining_nodes)>0:
        print('db size',remain_db.shape)
        db=remain_db.loc[(remain_db['root'] == start) &\
                (remain_db['root_ik_number'] == start_ik)]

        possible_nodes=list(db.goal.unique())
        if len(possible_nodes)==0:
            return float('inf'),sequence
        next =random.choice(possible_nodes)
        db=db.loc[(db['goal'] == next)]

        remaining_nodes.remove(next)
        while True:
            next_ik = random.randrange(ik_number[start])
            element=db.loc[(db['goal_ik_number'] == next_ik)]
            if (len(element)>0):
                break

        cost=element.iloc[0]['cost']
        total_cost+=cost
        if (total_cost>best_cost):  # abort this solution because is worst than the best one
            return float('inf'),sequence
        sequence.append({'node': next,'ik': float(next_ik)})
        start=next
        start_ik=next_ik
        remain_db=remain_db.loc[(remain_db['goal'] != start)]

    return total_cost, sequence

def main():
    rospy.init_node('min_travel')


    rospack = rospkg.RosPack()
    yaml_path=rospack.get_path('leonardo_launchers')
    pingInfoFilePath = yaml_path+"/config/costmap.ftr";
    cost_db = pd.read_feather(pingInfoFilePath, columns=None, use_threads=True);

    nodes=list(cost_db.root.unique())
    ik_number={}
    for n in nodes:
        ik_number[n]=int(1+cost_db.loc[cost_db['root'] == n]['root_ik_number'].max())

    best_cost=float('inf')
    best_sequence=[]
    for idx in range(0,100):
        print("cycle ",idx)
        travel_cost, travel_sequence=genClosestFirstTravel(nodes,ik_number,cost_db,best_cost,best_sequence)
        if travel_cost<best_cost:
            best_cost=travel_cost
            best_sequence=travel_sequence
            print('- improve cost to',best_cost)
            break


    column = cost_db['cost']
    min_cost = cost_db['cost'].min()

    count = column[column > best_cost-(min_cost*(len(nodes)-2))].count()
    print('row with too much cost',count,'over', cost_db.shape[0])
    filter_db=cost_db.loc[cost_db['cost']<(best_cost-(min_cost*(len(nodes)-2)))]

    for idx in range(0,10):
        print("cycle ",idx)
        travel_cost, travel_sequence=genClosestFirstTravel(nodes,ik_number,filter_db,best_cost,best_sequence)
        if travel_cost<best_cost:
            best_cost=travel_cost
            best_sequence=travel_sequence
            print('- improve cost to',best_cost)
            break

    print(best_sequence)
    print(best_cost)
    travel=list(best_sequence)

    for idx in range(1,len(travel)):
        print('- %s/iksol%d ==> %s/iksol%d' % (travel[idx-1]['node'], travel[idx-1]['ik'], travel[idx]['node'], travel[idx]['ik']))
    rospy.set_param("/precompute_trees/travel",travel)

    with open(yaml_path+'/config/travel.yaml', 'w') as file:
        documents = yaml.dump(travel, file)
if __name__ == "__main__":
    main()
