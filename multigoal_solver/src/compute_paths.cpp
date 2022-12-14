#include <ros/ros.h>

#include <graph_core/moveit_collision_checker.h>
#include <graph_core/parallel_moveit_collision_checker.h>
#include <graph_core/sampler.h>
#include <graph_core/metrics.h>
#include <graph_core/solvers/rrt.h>
#include <graph_core/graph/tree.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <ik_solver_msgs/GetIk.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_subtree");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ros::NodeHandle solver_nh("~/solver");



  std::string group_name = "manipulator";
  if (!pnh.getParam("group_name",group_name))
  {
    ROS_ERROR("%s/group_name is not defined",pnh.getNamespace().c_str());
    return 0;
  }
  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr           kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScenePtr     planning_scene = std::make_shared<planning_scene::PlanningScene>(kinematic_model);
  std::vector<std::string> joint_names = kinematic_model->getJointModelGroup(group_name)->getActiveJointModelNames();
  unsigned int dof = joint_names.size();
  Eigen::VectorXd lb(dof);
  Eigen::VectorXd ub(dof);
  pathplan::SamplerPtr sampler=std::make_shared<pathplan::InformedSampler>(lb,ub,lb,ub);
  double rewire_radius= 1.1 * std::pow(2 * (1.0 + 1.0 / dof) * (sampler->getSpecificVolume()), 1.0 / dof);

  for (unsigned int idx = 0; idx < dof; idx++)
  {
    const robot_model::VariableBounds& bounds = kinematic_model->getVariableBounds(joint_names.at(idx));
    if (bounds.position_bounded_)
    {
      lb(idx) = bounds.min_position_;
      ub(idx) = bounds.max_position_;
    }
  }


  int num_threads =pnh.param("number_of_threads",5);
  double steps=pnh.param("collision_steps",0.01);
  double maximum_distance=pnh.param("maximum_distance",0.01);

  pathplan::CollisionCheckerPtr checker = std::make_shared<pathplan::ParallelMoveitCollisionChecker>(planning_scene, group_name,num_threads,steps);
  pathplan::MetricsPtr metrics=std::make_shared<pathplan::Metrics>();
  Eigen::VectorXd q(dof);
  q.setZero();



  int number_of_poi;
  if (!pnh.getParam("/goals/number_of_poi",number_of_poi))
  {
    ROS_ERROR("/goals/number_of_poi is not defined");
    return 0;
  }

  std::string node_prefix_name;
  if (!pnh.getParam("/goals/node_prefix_name",node_prefix_name))
  {
    ROS_ERROR("/goals/node_prefix_name is not defined");
    return 0;
  }
  std::vector<std::string> tf_list;
  for (int idx=0;idx<number_of_poi;idx++)
  {
    tf_list.push_back(node_prefix_name+std::to_string(idx));
  }

  std::string tree_namespace;
  if (!pnh.getParam("tree_namespace",tree_namespace))
  {
    ROS_ERROR("%s/tree_namespace is not defined",pnh.getNamespace().c_str());
    return 0;
  }


  ros::NodeHandle tree_nh(tree_namespace);

  XmlRpc::XmlRpcValue result;
  int idx=0;

  for (std::string& tf_name: tf_list)
  {
    int nsol;
    if (!tree_nh.getParam("/goals/"+tf_name+"/number_of_ik",nsol))
    {
      ROS_ERROR("%s is unable to load trees",pnh.getNamespace().c_str());
      return 0;
    }
    for (int isol=0;isol<nsol;isol++)
    {
      XmlRpc::XmlRpcValue p;
      std::string tree_name="/goals/"+tf_name+"/iksol"+std::to_string(isol);
      ROS_INFO("loading tree from ik_solution%d of %s",isol,tf_name.c_str());

      if (!nh.getParam(tree_name+"/tree",p))
      {
        ROS_ERROR("%s is unable to load trees",pnh.getNamespace().c_str());
        return 0;
      }
      pathplan::TreePtr tree=pathplan::Tree::fromXmlRpcValue(p,maximum_distance,checker,metrics,true);

      for (std::string& destination: tf_list)
      {
        if (!tf_name.compare(destination))
          continue;
        int nsol_dest;
        if (!tree_nh.getParam("/goals/"+destination+"/number_of_ik",nsol_dest))
        {
          ROS_ERROR("%s is unable to load trees",pnh.getNamespace().c_str());
          return 0;
        }
        for (int isol_dest=0;isol_dest<nsol_dest;isol_dest++)
        {

          std::string dest_name="/goals/"+destination+"/iksol"+std::to_string(isol_dest);
          ROS_INFO("Destination %s",dest_name.c_str());
          std::vector<double> dest_configuration;
          if (!tree_nh.getParam(dest_name+"/root",dest_configuration))
          {
            ROS_ERROR("%s is unable to load destination",pnh.getNamespace().c_str());
            return 0;
          }
          Eigen::VectorXd q=Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(dest_configuration.data(),
                                                                          dest_configuration.size());
          pathplan::NodePtr goal = std::make_shared<pathplan::Node>(q);
          pathplan::PathPtr solution;
          double cost=std::numeric_limits<double>::infinity();
          pathplan::NodePtr new_node;

          bool connected=false;
          connected=tree->connectToNode(goal,new_node);

          if (connected)
          {
            solution=std::make_shared<pathplan::Path>(tree->getConnectionToNode(goal),metrics,checker);
            cost=solution->cost();
          }
          else
          {
            pathplan::SamplerPtr sampler=std::make_shared<pathplan::InformedSampler>(tree->getRoot()->getConfiguration(),goal->getConfiguration(),lb,ub);

            pathplan::RRT solver(metrics,checker,sampler);
            solver.config(solver_nh);
            solver.addStartTree(tree);
            solver.addGoal(goal);

            if (!solver.solve(solution))
            {
              ROS_WARN("Unable to solve from %s/iksol%d to %s/iksol%d",tf_name.c_str(),isol,destination.c_str(),isol_dest);
            }
            else
            {
              cost=solution->cost();
            }
          }

          if (cost<std::numeric_limits<double>::infinity())
          {
            XmlRpc::XmlRpcValue path=solution->toXmlRpcValue();
            tree_nh.setParam(tree_name+"/path/"+dest_name,path);
          }
          ROS_INFO("Solution from %s/iksol%d to %s/iksol%d, cost=%f",tf_name.c_str(),isol,destination.c_str(),isol_dest,cost);
          XmlRpc::XmlRpcValue r;
          r["root"]=tf_name;
          r["goal"]=destination;
          r["root_ik_number"]=isol;
          r["goal_ik_number"]=isol_dest;
          r["cost"]=cost;
          result[idx++]=r;
        }
      }

      pnh.setParam(tree_name+"/tree",tree->toXmlRpcValue());

    }
  }

  pnh.setParam("cost_map",result);

  ROS_INFO("%s complete the task",pnh.getNamespace().c_str());
  return 0;
}
