#include <ros/ros.h>

#include <graph_core/moveit_collision_checker.h>
#include <graph_core/parallel_moveit_collision_checker.h>
#include <graph_core/sampler.h>
#include <graph_core/metrics.h>
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
  for (unsigned int idx = 0; idx < dof; idx++)
  {
    const robot_model::VariableBounds& bounds = kinematic_model->getVariableBounds(joint_names.at(idx));
    if (bounds.position_bounded_)
    {
      lb(idx) = bounds.min_position_;
      ub(idx) = bounds.max_position_;
      ROS_INFO("joint %s has bounds = [%f,%f]",joint_names.at(idx).c_str(),lb(idx),ub(idx));
    }
  }


  int num_threads =pnh.param("number_of_threads",5);
  double steps=pnh.param("collision_steps",0.01);
  double maximum_distance=pnh.param("maximum_distance",0.01);

  pathplan::CollisionCheckerPtr checker = std::make_shared<pathplan::ParallelMoveitCollisionChecker>(planning_scene, group_name,num_threads,steps);
  pathplan::MetricsPtr metrics=std::make_shared<pathplan::Metrics>();
  pathplan::SamplerPtr sampler=std::make_shared<pathplan::InformedSampler>(lb,ub,lb,ub);
  double rewire_radius= 1.1 * std::pow(2 * (1.0 + 1.0 / dof) * (sampler->getSpecificVolume()), 1.0 / dof);
  Eigen::VectorXd q(dof);
  q.setZero();

  int number_of_poi;
  if (!nh.getParam("/goals/number_of_poi",number_of_poi))
  {
    ROS_ERROR("/goals/number_of_poi is not defined");
    return 0;
  }

  std::string node_prefix_name;
  if (!nh.getParam("/goals/node_prefix_name",node_prefix_name))
  {
    ROS_ERROR("/goals/node_prefix_name is not defined");
    return 0;
  }
  std::vector<std::string> tf_list;
  for (int idx=0;idx<number_of_poi;idx++)
  {
    tf_list.push_back(node_prefix_name+std::to_string(idx));
  }

  int number_of_nodes;
  if (!pnh.getParam("number_of_nodes",number_of_nodes))
  {
    ROS_ERROR("%s/number_of_nodes is not defined",pnh.getNamespace().c_str());
    return 0;
  }
  int stall_iterations;
  if (!pnh.getParam("stall_iterations",stall_iterations))
  {
    ROS_ERROR("%s/stall_iterations is not defined",pnh.getNamespace().c_str());
    return 0;
  }
  int max_number_of_solutions;
  if (!pnh.getParam("max_number_of_solutions",max_number_of_solutions))
  {
    ROS_ERROR("%s/max_number_of_solutions is not defined",pnh.getNamespace().c_str());
    return 0;
  }

  for (std::string& tf_name: tf_list)
  {

    int number_ik;
    if (!nh.getParam("/goals/"+tf_name+"/number_of_ik",number_ik))
    {
      ROS_ERROR_STREAM("unable to read parameter "<< "/goals/"+tf_name+"/number_of_ik");
      return 0;
    }

    for (int isol=0;isol<number_ik;isol++)
    {
      std::string tree_name="/goals/"+tf_name+"/iksol"+std::to_string(isol);
      std::vector<double> iksol;

      if (!nh.getParam(tree_name+"/root",iksol))
      {
        ROS_ERROR_STREAM("unable to read parameter "<< tree_name+"/root");
        return 0;
      }

      q=Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(iksol.data(), iksol.size());

      pathplan::NodePtr root = std::make_shared<pathplan::Node>(q);

      if (!checker->check(q))
      {
        ROS_FATAL("this should not happen");
        return 0;
      }
      pathplan::TreePtr tree=std::make_shared<pathplan::Tree>(root,maximum_distance,checker,metrics);

      pathplan::NodePtr new_node;
      ROS_DEBUG("growing tree from ik_solution %d of %s",isol,tf_name.c_str());
      ros::WallTime t0=ros::WallTime::now();
      int iter=0;
      while ((int)tree->getNumberOfNodes()<number_of_nodes)
      {
        if (not ros::ok())
        {
          ROS_WARN("%s has been stopped because ros is not ok",pnh.getNamespace().c_str());
          return 0;
        }
        if (iter++>1e5)
          break;
        q=sampler->sample();

        tree->rewire(q,rewire_radius,new_node);
        tree->extend(q,new_node);
        rewire_radius = 1.1 * std::pow(2 * (1.0 + 1.0 / dof) * (sampler->getSpecificVolume()), 1.0 / dof);
      }
      ros::WallTime t1=ros::WallTime::now();
      ROS_INFO("saving tree from  ik_solution %d of %s (computed in %f ms, size %u)",isol,tf_name.c_str(),(t1-t0).toSec()*1e3,tree->getNumberOfNodes());
      pnh.setParam(tree_name+"/tree",tree->toXmlRpcValue());


    }

  }

  ROS_INFO("%s complete the task",pnh.getNamespace().c_str());
  return 0;
}
