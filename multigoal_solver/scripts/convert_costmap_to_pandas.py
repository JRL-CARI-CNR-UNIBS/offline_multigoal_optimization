#!/usr/bin/env python3
import rospy
import rosservice
import math
import numpy as np
import pandas as pd

import feather
#sudo apt install python3-feather-format
#pip3 instal pyarrow

def client():
    rospy.init_node('convert_to_pandas')

    cost_map=rospy.get_param("/precompute_trees/cost_map")

    root=[]
    root_ik_number=[]
    goal=[]
    goal_ik_number=[]
    cost=[]
    for row in cost_map:
        root.append(row["root"])
        root_ik_number.append(row["root_ik_number"])
        goal.append(row["goal"])
        goal_ik_number.append(row["goal_ik_number"])
        cost.append(row["cost"])

    d = {'root': root, 'root_ik_number': root_ik_number, 'goal': goal, 'goal_ik_number': goal_ik_number, 'cost': cost}
    df = pd.DataFrame(data=d)

    pingInfoFilePath = "./costmap.ftr";
    df.to_feather(pingInfoFilePath);
    print(df)
if __name__ == "__main__":
    client()
