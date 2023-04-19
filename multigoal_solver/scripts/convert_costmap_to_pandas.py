#!/usr/bin/env python3
import os.path

import rospy
import pandas as pd
import rospkg
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest


class Converter2Pandas:
    def __init__(self):
        rospy.init_node('convert_to_pandas')

        self.service = rospy.Service('/convert_to_pandas', Trigger, self.cb_convert_to_pandas)

    def cb_convert_to_pandas(self, req: TriggerRequest) -> TriggerResponse:
        rospack = rospkg.RosPack()
        yaml_path = rospack.get_path('leonardo_launchers')
        cost_map = rospy.get_param("/precompute_trees/cost_map")

        root = [row['root'] for row in cost_map]
        root_ik_number = [row['root_ik_number'] for row in cost_map]
        goal = [row['goal'] for row in cost_map]
        goal_ik_number = [row['goal_ik_number'] for row in cost_map]
        cost = [row['cost'] for row in cost_map]

        d = {'root': root,
             'root_ik_number': root_ik_number,
             'goal': goal,
             'goal_ik_number': goal_ik_number,
             'cost': cost}
        df = pd.DataFrame(data=d)

        pingInfoFilePath = os.path.join(yaml_path , 'config/costmap.ftr')
        df.to_feather(pingInfoFilePath)
        print(df)

        resp = TriggerResponse()
        resp.success = True
        resp.message = f'Dtaframe saved in {pingInfoFilePath}'

        return resp


def main():
    node = Converter2Pandas()
    rospy.spin()
    return


if __name__ == "__main__":
    main()
