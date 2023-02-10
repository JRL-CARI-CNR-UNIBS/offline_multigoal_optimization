#include <ros/ros.h>

#include <graph_core/moveit_collision_checker.h>
#include <graph_core/parallel_moveit_collision_checker.h>
#include <graph_core/sampler.h>
#include <graph_core/metrics.h>
#include <graph_core/graph/tree.h>
#include <graph_core/solvers/rrt_star.h>
#include <graph_core/solvers/path_solver.h>
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

  double exploring_coef=5.0;

  std::string group_name = "manipulator";
  if (!pnh.getParam("group_name",group_name))
  {
    ROS_ERROR("%s/group_name is not defined",pnh.getNamespace().c_str());
    return 0;
  }

  double max_time;
  if (!solver_nh.getParam("solver_max_time",max_time))
  {
    ROS_WARN("%s/solver_max_time is not defined, set 5.0 second",pnh.getNamespace().c_str());
    max_time=5.0;
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

  std::map<std::pair<std::string,std::string>,double> best_cost_map;
  std::map<std::string,double> best_cost_from_goal;

  XmlRpc::XmlRpcValue result;
  int idx_result=0;

  for (std::string& tf_name: tf_list)
  {
    best_cost_from_goal.insert(std::pair<std::string,double>(tf_name,std::numeric_limits<double>::infinity()));
    double& best_cost_from_this_goal=best_cost_from_goal.at(tf_name);

    std::multimap<double,std::tuple<std::string,Eigen::VectorXd,double*,int>> ordered_goal;

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
        continue;
      }

      pathplan::TreePtr tree=std::make_shared<pathplan::Tree>(root,maximum_distance,checker,metrics);

      pathplan::NodePtr new_node;
      ROS_DEBUG("growing tree from ik_solution %d of %s",isol,tf_name.c_str());
      ros::WallTime t0=ros::WallTime::now();

      for (std::string& tf_name2: tf_list)
      {
        if (!tf_name2.compare(tf_name))
          continue;
        std::pair<std::string,std::string> p(tf_name,tf_name2);
        if (best_cost_map.count(p)==0)
        {
          best_cost_map.insert(std::pair<std::pair<std::string,std::string>,double>(p,std::numeric_limits<double>::infinity()));
        }
        for (int isol2=0;isol2<number_ik;isol2++)
        {
          std::string tree_name2="/goals/"+tf_name2+"/iksol"+std::to_string(isol2);
          if (!nh.getParam(tree_name2+"/root",iksol))
          {
            ROS_ERROR_STREAM("unable to read parameter "<< tree_name2+"/root");
            return 0;
          }

          q=Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(iksol.data(), iksol.size());
          double dist=metrics->utopia(tree->getRoot()->getConfiguration(),q);
          std::tuple<std::string,Eigen::VectorXd,double*,int> t(tf_name2,q,&best_cost_map.at(std::pair<std::string,std::string>(tf_name,tf_name2)),isol2);
          std::pair<double,std::tuple<std::string,Eigen::VectorXd,double*,int>> p(dist,t);
          ordered_goal.insert(p);
        }
      }

      ROS_INFO("goal distance are in range [%f,%f]",ordered_goal.begin()->first,(--ordered_goal.end())->first);


      for (const std::pair<double,std::tuple<std::string,Eigen::VectorXd,double*,int>>& p: ordered_goal)
      {
        double* best_goal_from_tree_to_tree2=std::get<2>(p.second);
        std::string tf_name2=std::get<0>(p.second);
        int isol_dest=std::get<3>(p.second);
        if (!ros::ok())
          return 0;

        std::pair<std::string,std::string> reverse_node_pair(tf_name2,tf_name);
        if (best_cost_map.count(reverse_node_pair)>0)
        {
          if (best_cost_map.at(reverse_node_pair)<*best_goal_from_tree_to_tree2)
            *best_goal_from_tree_to_tree2=best_cost_map.at(reverse_node_pair);
        }
        q=std::get<1>(p.second);

        pathplan::NodePtr goal = std::make_shared<pathplan::Node>(q);

        pathplan::SamplerPtr sampler=std::make_shared<pathplan::InformedSampler>(tree->getRoot()->getConfiguration(),
                                                                                 goal->getConfiguration(),
                                                                                 lb,
                                                                                 ub);
        if (sampler->getFociiDistance()>*best_goal_from_tree_to_tree2)
        {
          ROS_DEBUG("%s/iksol%d null informed set.",tf_name.c_str(),isol);
          continue;
        }
        if (sampler->getFociiDistance()>exploring_coef*best_cost_from_this_goal)
          continue;

        pathplan::RRT solver(metrics,checker,sampler);
        solver.config(solver_nh);
        solver.addStartTree(tree);
        solver.addGoal(goal);

        ros::WallTime tsolver=ros::WallTime::now();
        pathplan::PathPtr solution;
        bool solved=false;

        while ((ros::WallTime::now()-tsolver).toSec()<max_time)
        {
          if (solver.update(solution))
          {
            solved=true;
            break;
          }
        }

        if (solved)
        {
          solution->setTree(tree);
          pathplan::PathLocalOptimizer path_opt(checker,metrics);
          path_opt.setPath(solution);

          if (path_opt.solve(solution))
          {
            if (*best_goal_from_tree_to_tree2>solution->cost())
            {
              *best_goal_from_tree_to_tree2=solution->cost();
              sampler->setCost(*best_goal_from_tree_to_tree2);
              ROS_INFO("improve best cost of tree %s->%s to %f",tf_name.c_str(),tf_name2.c_str(),*best_goal_from_tree_to_tree2);
            }
          }
        }

        if (solved && (int)tree->getNumberOfNodes()<number_of_nodes)
        {
          pathplan::RRTStar rrtstar_solver(metrics,checker,sampler);
          rrtstar_solver.config(solver_nh);
          rrtstar_solver.addStartTree(tree);
          rrtstar_solver.addGoal(goal);

          if (*best_goal_from_tree_to_tree2>solution->cost())
          {
            *best_goal_from_tree_to_tree2=solution->cost();
            sampler->setCost(*best_goal_from_tree_to_tree2);
          }

          while ((ros::WallTime::now()-tsolver).toSec()<max_time)
          {
            if (rrtstar_solver.update(solution))
            {
              solution->setTree(tree);
              if (*best_goal_from_tree_to_tree2>solution->cost())
              {
                ROS_INFO("improve best cost of tree %s->%s to %f",tf_name.c_str(),tf_name2.c_str(),*best_goal_from_tree_to_tree2);
                *best_goal_from_tree_to_tree2=solution->cost();
                sampler->setCost(*best_goal_from_tree_to_tree2);
              }
            }
            if (rrtstar_solver.completed())
              break;
          }


        }

        if (solved)
        {
          double cost=solution->cost();
          best_cost_from_this_goal=std::min(cost,best_cost_from_this_goal);


          XmlRpc::XmlRpcValue path=solution->toXmlRpcValue();
          pnh.setParam(tree_name+"/path/"+tf_name2+"/iksol"+std::to_string(isol_dest),path);
          ROS_INFO("Solution from %s/iksol%d to %s/iksol%d, cost=%f",tf_name.c_str(),isol,tf_name2.c_str(),isol_dest,cost);
          XmlRpc::XmlRpcValue r;
          r["root"]=tf_name;
          r["goal"]=tf_name2;
          r["root_ik_number"]=isol;
          r["goal_ik_number"]=isol_dest;
          r["cost"]=cost;
          result[idx_result++]=r;
        }
      }

      pnh.setParam(tree_name+"/tree",tree->toXmlRpcValue());
      std::string file_name=tree_name+".xml";
      std::replace( file_name.begin(), file_name.end(), '/', '-');
      tree->toXmlFile(file_name);
      ROS_INFO("grown tree from ik_solution %d of %s in %f seconds, number of nodes %d",isol,tf_name.c_str(),(ros::WallTime::now()-t0).toSec(),tree->getNumberOfNodes());
    }
  }

  pnh.setParam("cost_map",result);
  ROS_INFO("%s complete the task",pnh.getNamespace().c_str());
  return 0;
}
