#include <ros/ros.h>

#include <graph_core/moveit_collision_checker.h>
#include <graph_core/parallel_moveit_collision_checker.h>
#include <graph_core/sampler.h>
#include <graph_core/metrics.h>
#include <graph_core/solvers/path_solver.h>
#include <graph_core/graph/tree.h>
#include <graph_core/graph/subtree.h>
#include <graph_core/solvers/rrt.h>
#include <graph_core/solvers/multigoal.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <ik_solver_msgs/GetIk.h>
#include <ik_solver_msgs/GetIkArray.h>
#include <ik_solver_msgs/GetBound.h>
#include <std_srvs/Trigger.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <limits>
#include "ik_solver_msgs/IkTarget.h"

void pointCloudCb(const sensor_msgs::PointCloud2ConstPtr& msg, sensor_msgs::PointCloud2Ptr& pc)
{
  pc.reset(new sensor_msgs::PointCloud2(*msg));
}

bool pathCb(std_srvs::TriggerRequest& req, std_srvs::TriggerResponse& res)
{
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ros::NodeHandle solver_nh("~/solver");


  std::string group_name = "manipulator";
  if (!pnh.getParam("group_name", group_name))
  {
    res.message =  pnh.getNamespace() + "/group_name is not defined";
    ROS_ERROR("%s", res.message.c_str());
    res.success = false;
    return true;
  }

  double max_path_step;
  if (!pnh.getParam("max_path_step", max_path_step))
  {
    res.message =  pnh.getNamespace() + "/max_path_step is not defined";
    ROS_ERROR("%s", res.message.c_str());
    res.success = false;
    return true;
  }
  if (max_path_step<=0)
  {
    res.message =  pnh.getNamespace() + "/max_path_step must be positive";
    ROS_ERROR("%s", res.message.c_str());
    res.success = false;
    return true;
  }


  ros::ServiceClient ps_client = nh.serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");
  ps_client.waitForExistence();
  moveit_msgs::GetPlanningScene ps_srv;
  if (!ps_client.call(ps_srv))
  {
    res.message =  "Error on  get_planning_scene srv not ok";
    ROS_ERROR("%s", res.message.c_str());
    res.success = false;
    return true;
  }

  std::vector<double> w;
  if (!pnh.getParam("weight", w))
  {
    res.success = false;
    res.message =  pnh.getNamespace() + "/weight is not defined";
    ROS_ERROR("%s", res.message.c_str());
    return true;
  }

  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScenePtr planning_scene = std::make_shared<planning_scene::PlanningScene>(kinematic_model);
  std::vector<std::string> joint_names = kinematic_model->getJointModelGroup(group_name)->getActiveJointModelNames();
  unsigned int dof = joint_names.size();

  int num_threads = pnh.param("number_of_threads", 5);
  double steps = pnh.param("collision_steps", 0.01);
  double maximum_distance = pnh.param("maximum_distance", 0.01);
  double max_computation_time = pnh.param("online_max_time", 5.0);



  std::map<std::string,double> online_max_joint_elongation;
  if (!pnh.getParam("online_max_joint_elongation",online_max_joint_elongation))
  {
    online_max_joint_elongation.clear();
  }


  bool enable_approach = pnh.param("enable_approach", true);

  pathplan::CollisionCheckerPtr checker =
      std::make_shared<pathplan::MoveitCollisionChecker>(planning_scene, group_name, maximum_distance);
  checker->setPlanningSceneMsg(ps_srv.response.scene);
  pathplan::MetricsPtr metrics = std::make_shared<pathplan::Metrics>();

  XmlRpc::XmlRpcValue travel;

  if (!pnh.getParam("travel", travel))
  {
    res.message =  pnh.getNamespace() + "/travel is not defined";
    ROS_ERROR("%s", res.message.c_str());
    res.success = false;
    return true;
  }


  sensor_msgs::PointCloud2Ptr pc;

  ros::Subscriber pc_sub =
      nh.subscribe<sensor_msgs::PointCloud2>("/point_cloud2", 100, boost::bind(pointCloudCb, _1, boost::ref(pc)));

  std::string tool_name;
  ros::ServiceClient ik_client;
  ros::ServiceClient bound_client;
  if (!nh.getParam("/tool_name", tool_name))
  {
    ik_client = nh.serviceClient<ik_solver_msgs::GetIkArray>("/ik_solver/get_ik_array");
    bound_client = nh.serviceClient<ik_solver_msgs::GetBound>("ik_solver/get_bounds");
  }
  else
  {
    ik_client = nh.serviceClient<ik_solver_msgs::GetIkArray>("/" + tool_name + "_ik_solver/get_ik_array");
    bound_client = nh.serviceClient<ik_solver_msgs::GetBound>("/" + tool_name + "_ik_solver/get_bounds");
  }

  if (!bound_client.waitForExistence(ros::Duration(20)))
  {
    res.message =  "Timeout on server " + bound_client.getService();
    ROS_ERROR("%s", res.message.c_str());
    res.success = false;
    return true;
  }

  ik_solver_msgs::GetBoundRequest bound_req;
  ik_solver_msgs::GetBoundResponse bound_res;
  if (!bound_client.call(bound_req,bound_res))
  {

    res.message =  "Unable to call service " + bound_client.getService();
    ROS_ERROR("%s", res.message.c_str());
    res.success = false;
    return true;
  }

  joint_names=bound_res.joint_names;
  Eigen::VectorXd lb(joint_names.size());
  Eigen::VectorXd ub(joint_names.size());
  Eigen::VectorXd elongation(joint_names.size());

  for (size_t iax=0; iax < bound_res.joint_names.size(); iax++)
  {
    if (online_max_joint_elongation.count(bound_res.joint_names.at(iax))>0)
    {
      elongation(iax)=online_max_joint_elongation.at(bound_res.joint_names.at(iax));
    }
    else
    {
      elongation(iax)=std::nan("1"); //not set
    }

    std::vector< ::ik_solver_msgs::JointRange>::const_iterator im = std::min_element(bound_res.boundaries.at(iax).joint_ranges.begin(), bound_res.boundaries.at(iax).joint_ranges.end(), [](const ik_solver_msgs::JointRange& lhs, const ik_solver_msgs::JointRange & rhs){ return lhs.lower_bound > rhs.lower_bound;});
    lb(iax)= (im == bound_res.boundaries.at(iax).joint_ranges.end() ? - std::numeric_limits<double>::infinity() : im->lower_bound);

    std::vector< ::ik_solver_msgs::JointRange>::const_iterator iM = std::max_element(bound_res.boundaries.at(iax).joint_ranges.begin(), bound_res.boundaries.at(iax).joint_ranges.end(), [](const ik_solver_msgs::JointRange& lhs, const ik_solver_msgs::JointRange & rhs){ return lhs.lower_bound < rhs.lower_bound;});
    ub(iax)=(im == bound_res.boundaries.at(iax).joint_ranges.end() ? std::numeric_limits<double>::infinity() : im->upper_bound);
  }

  ros::Publisher failed_poses_pub = nh.advertise<geometry_msgs::PoseArray>("fail_poses", 10, true);
  ros::Publisher no_feasible_ik_poses_pub = nh.advertise<geometry_msgs::PoseArray>("no_feasible_ik_poses", 10, true);
  ros::Publisher no_ik_poses_pub = nh.advertise<geometry_msgs::PoseArray>("no_ik_poses", 10, true);

  ROS_INFO("%s is waiting for the point cloud", pnh.getNamespace().c_str());

  ros::Rate lp(50);
  while (ros::ok())
  {
    ros::spinOnce();
    if (pc)
    {
      break;
    }
    lp.sleep();
  }

  int data_size = pc->data.size() / (sizeof(float));

  std::vector<float> data(data_size);
  memcpy(&(data.at(0)), &(pc->data.at(0)), pc->data.size());

  int n_points = data_size / pc->fields.size();
  geometry_msgs::PoseArray all_poses;
  all_poses.header.frame_id = pc->header.frame_id;
  all_poses.poses.resize(n_points);
  std::vector<int> group(n_points);

  int idx = 0;
  for (int ip = 0; ip < n_points; ip++)
  {
    geometry_msgs::Pose& p = all_poses.poses.at(ip);

    p.position.x = data.at(idx++);
    p.position.y = data.at(idx++);
    p.position.z = data.at(idx++);
    p.orientation.x = data.at(idx++);
    p.orientation.y = data.at(idx++);
    p.orientation.z = data.at(idx++);
    p.orientation.w = data.at(idx++);
    group.at(ip) = data.at(idx++);
  }

  Eigen::VectorXd q;
  Eigen::VectorXd last_q;
  geometry_msgs::PoseArray fail_poses;
  fail_poses.header.frame_id = pc->header.frame_id;

  geometry_msgs::PoseArray no_feasible_ik_poses;
  no_feasible_ik_poses.header.frame_id = pc->header.frame_id;

  geometry_msgs::PoseArray no_ik_poses;
  no_ik_poses.header.frame_id = pc->header.frame_id;
  no_ik_poses.poses.clear();

  bool first_node = true;
  bool first_time = true;
  pathplan::TreePtr tree;
  pathplan::NodePtr last_node;
  pathplan::NodePtr new_node;

  std::vector<pathplan::ConnectionPtr> connections;
  std::vector<int> order_pose_number;
  order_pose_number.push_back(-10);  // n connections have n+1 nodes

  XmlRpc::XmlRpcValue configurations;
  std::vector<int> configurations_number;
  int configurations_size = 0;

  ros::param::set("/complete/number_of_fail_poses", 0);
  ros::param::set("/complete/number_of_no_feasible_ik_poses", 0);
  ros::param::set("/complete/number_of_no_ik_poses", 0);
  

  for (int inode = 0; inode < travel.size(); inode++)
  {

    std::string node = travel[inode]["node"];

    int ik_sol;
    if (travel[inode]["ik"].getType()==travel[inode]["ik"].TypeInt)
      ik_sol = static_cast<int>(travel[inode]["ik"]);
    else
      ik_sol = (int)static_cast<double>(travel[inode]["ik"]);

    ik_solver_msgs::GetIkArrayRequest ik_req;
    ik_solver_msgs::GetIkArrayResponse ik_res;
    
    ik_req.seed_joint_names = joint_names;
    ik_solver_msgs::Configuration seed;

    std::string s = node;
    s.erase(std::remove_if(std::begin(s), std::end(s), [](char ch) { return !std::isdigit(ch); }), s.end());
    int igroup = std::stoi(s);
    /* */ ROS_DEBUG("node %s ik=%d, group=%d", node.c_str(), ik_sol, igroup);

    std::string tree_name = "/goals/" + node + "/iksol" + std::to_string(ik_sol);
    std::vector<double> iksol;

    if (!nh.getParam(tree_name + "/root", iksol))
    {
      res.message =  "unable to read parameter " + tree_name + "/root";
      ROS_ERROR("%s", res.message.c_str());
      res.success = false;
      return true;
    }

    seed.configuration = iksol;
    

    q = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(iksol.data(), iksol.size());
    Eigen::VectorXd approach = q;
    Eigen::MatrixXd weight(q.size(), q.size());
    weight.setIdentity();
    for (int iax = 0; iax < q.size(); iax++)
      weight(iax, iax) = w.at(iax);

    if (first_node)
    {
      last_q = q;
      pathplan::NodePtr root = std::make_shared<pathplan::Node>(q);
      last_node = root;

      if (!checker->check(q))
      {
        res.message = tree_name + ": the root node is unreachable and this should not happen..";
        ROS_ERROR("%s", res.message.c_str());
        res.success = false;
        return true;
      }

      tree = std::make_shared<pathplan::Tree>(root, maximum_distance, checker, metrics);
      first_node = false;
    }
    else
    {
      if (checker->checkPath(last_node->getConfiguration(), approach))
      {
        last_q = approach;
        new_node = std::make_shared<pathplan::Node>(approach);
        pathplan::ConnectionPtr conn = std::make_shared<pathplan::Connection>(last_node, new_node);
        conn->add();
        conn->setCost(metrics->cost(last_node->getConfiguration(), approach));
        connections.push_back(conn);
        order_pose_number.push_back(-10);
        tree->addNode(new_node);
        last_node = new_node;
        /* */ ROS_DEBUG("connect with next keypoint");
      }
      else
      {
        pathplan::SubtreePtr subtree=std::make_shared<pathplan::Subtree>(tree,last_node);

        Eigen::VectorXd actual_lb=lb;
        Eigen::VectorXd actual_ub=ub;
        for (unsigned int iax=0;iax<lb.size();iax++)
        {
          if (!std::isnan(elongation(iax)) && elongation(iax)>=0)
          {
            double min_value=std::min(last_node->getConfiguration()(iax),approach(iax));
            double max_value=std::max(last_node->getConfiguration()(iax),approach(iax));
            actual_lb(iax)=std::max(lb(iax),min_value-elongation(iax));
            actual_ub(iax)=std::min(ub(iax),max_value+elongation(iax));
          }
        }
        pathplan::InformedSamplerPtr sampler=std::make_shared<pathplan::InformedSampler>(last_node->getConfiguration(),
                                                                                         approach,
                                                                                         actual_lb,
                                                                                         actual_ub);


        pathplan::NodePtr g = std::make_shared<pathplan::Node>(approach);
        pathplan::MultigoalSolver solver(metrics,checker,sampler);
        if(solver.config(solver_nh) && solver.addStartTree(subtree) && solver.addGoal(g))
        {
          pathplan::PathPtr solution;
          if (solver.solve(solution,1e5,max_computation_time))
          {
            last_q = approach;
            last_node = g;
            solution->setTree(tree);

            pathplan::PathLocalOptimizer path_opt(checker, metrics);
            path_opt.setPath(solution);
            path_opt.solve(solution,100000);
            std::vector<pathplan::ConnectionPtr> tmp_connections=solution->getConnections();


            for (size_t iconnection = 0; iconnection < tmp_connections.size(); iconnection++)
            {
              connections.push_back(tmp_connections.at(iconnection));
              order_pose_number.push_back(-10);
            }
          }
        }
      }
    }

    std::vector<int> pose_number;
    for (int ip = 0; ip < n_points; ip++)
    {
      if (group.at(ip) == igroup)
      {
        ik_solver_msgs::IkTarget target;
        target.pose.pose = all_poses.poses.at(ip);
        target.pose.header.frame_id = all_poses.header.frame_id;
        target.seeds = {seed};
        ik_req.targets.push_back(target);
        pose_number.push_back(ip);
      }
    }

    auto st = ros::Time::now();
    std::string hdr = "["+std::to_string(inode+1)+ (inode+1 % 10 == 1 ? "st" : inode+1 % 10 == 2 ? "nd" : "th") + " out of " + std::to_string(travel.size()) + " sub-areas]";
    ROS_WARN("%s >>>> Processing %zu poses ", hdr.c_str(), ik_req.targets.size());
    if (!ik_client.call(ik_req, ik_res))
    {
      res.message = pnh.getNamespace() + " unable to call '" + ik_client.getService() + "'";
      ROS_ERROR("%s", res.message.c_str());
      res.success = false;
      return true;
    }
    if(ik_req.targets.size() != ik_res.solutions.size())
    {
      res.message = pnh.getNamespace() + " Service '" + ik_client.getService() + "' returned a number of solutions different from expected. Req: " + std::to_string(ik_req.targets.size()) + " Res: " + std::to_string(ik_res.solutions.size());
      ROS_ERROR("%s", res.message.c_str());
      res.success = false;
      return true;
    }
    ROS_WARN("%s <<<< Processing %zu poses DT %fsec", hdr.c_str(), ik_req.targets.size(), (ros::Time::now()-st).toSec());

    st = ros::Time::now();
    ROS_WARN("%s >>>> Checking Collisions and Connect the %zu poses", hdr.c_str(), ik_res.solutions.size());
    for (size_t ip = 0; ip < ik_res.solutions.size(); ip++)
    {
      std::string _hdr = ">>>> " + hdr +" Pose " + std::to_string(ip) + " of " + std::to_string(ik_res.solutions.size()) + " (keypoint "+node+ ")";
      /* */ ROS_DEBUG(">>>> %s IK sols %zu", _hdr.c_str(), ik_res.solutions.at(ip).configurations.size());
      ik_solver_msgs::IkSolution& ik = ik_res.solutions.at(ip);
      bool connected = false;

      if (ik.configurations.empty())
      {
        /* */ ROS_DEBUG("%s IK did not found solution", _hdr.c_str());

        no_ik_poses.poses.push_back(ik_req.targets.at(ip).pose.pose);
        continue;
      }

      std::multimap<double, Eigen::VectorXd> ordered_configurations;
      for (ik_solver_msgs::Configuration& c : ik.configurations)
      {
        q = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(c.configuration.data(), c.configuration.size());

        if (checker->check(q))
        {
          double dist = std::sqrt(q.transpose() * weight * q);
          ordered_configurations.insert(std::pair<double, Eigen::VectorXd>(dist, q));
        }
      }

      if (ordered_configurations.empty())
      {
        /* */ ROS_DEBUG("%s IK found solution, but it is in collision ...", _hdr.c_str());
        no_feasible_ik_poses.poses.push_back(ik_req.targets.at(ip).pose.pose);
        continue;
      }
      if (!first_time)
      {
        for (const std::pair<double, Eigen::VectorXd>& p : ordered_configurations)
        {
          if (checker->checkPath(last_node->getConfiguration(), p.second))
          {
            last_q = p.second;
            connected = true;

            new_node = std::make_shared<pathplan::Node>(p.second);
            pathplan::ConnectionPtr conn = std::make_shared<pathplan::Connection>(last_node, new_node);
            conn->add();
            conn->setCost(metrics->cost(last_node->getConfiguration(), p.second));
            connections.push_back(conn);
            order_pose_number.push_back(pose_number.at(ip));
            tree->addNode(new_node);

            last_node = new_node;
            break;
          }
        }
      }
      if (!connected)
      {
        /* */ ROS_WARN("%s Try Connect", _hdr.c_str());
        pathplan::SubtreePtr subtree=std::make_shared<pathplan::Subtree>(tree,last_node);

        for (auto it = ordered_configurations.begin(); it != ordered_configurations.end(); ++it)
        {
          size_t iti = std::distance(ordered_configurations.begin(), it);
          std::string __hdr = "Try solving " +std::to_string(iti)+ " configuration of " + std::to_string(ordered_configurations.size());

          /* */ ROS_DEBUG(">>>> %s ", __hdr.c_str()); 
          const std::pair<double, Eigen::VectorXd>& p = *it;
          Eigen::VectorXd actual_lb=lb;
          Eigen::VectorXd actual_ub=ub;
          for (unsigned int iax=0;iax<lb.size();iax++)
          {
            if (!std::isnan(elongation(iax)) && elongation(iax)>=0)
            {
              double min_value=std::min(last_node->getConfiguration()(iax),p.second(iax));
              double max_value=std::max(last_node->getConfiguration()(iax),p.second(iax));
              actual_lb(iax)=std::max(lb(iax),min_value-elongation(iax));
              actual_ub(iax)=std::min(ub(iax),max_value+elongation(iax));
            }
          }

          pathplan::InformedSamplerPtr sampler=std::make_shared<pathplan::InformedSampler>(last_node->getConfiguration(),
                                                                                           p.second,
                                                                                           actual_lb,
                                                                                           actual_ub);


          pathplan::NodePtr g = std::make_shared<pathplan::Node>(p.second);
          pathplan::MultigoalSolver solver(metrics,checker,sampler);
          
          /* */ ROS_DEBUG(">>>> %s Solver Config", __hdr.c_str()); 
          if(solver.config(solver_nh) && solver.addStartTree(subtree) && solver.addGoal(g))
          {
            pathplan::PathPtr solution;

            /* */ ROS_DEBUG(">>>> %s Solve", __hdr.c_str()); 
            if (solver.solve(solution,1e5,max_computation_time))
            {
              last_q = p.second;
              connected = true;

              first_time = false;
              solution->setTree(tree);

              pathplan::PathLocalOptimizer path_opt(checker, metrics);
              path_opt.setPath(solution);
              path_opt.solve(solution,100000);
              std::vector<pathplan::ConnectionPtr> tmp_connections=solution->getConnections();

              last_node = g;

              for (size_t iconnection = 0; iconnection < tmp_connections.size(); iconnection++)
              {
                connections.push_back(tmp_connections.at(iconnection));
                order_pose_number.push_back(pose_number.at(ip));
              }
              break;
            }
          }
        }
      }

      if (!connected)
      {
        /* */ ROS_DEBUG(">>>> %s Not Connected ... ", _hdr.c_str()); 

        fail_poses.poses.push_back(ik_req.targets.at(ip).pose.pose);
      }
      else
      {
        XmlRpc::XmlRpcValue tmp_conf;
        for (int iax = 0; iax < last_q.size(); iax++)
        {
          tmp_conf[iax] = last_q(iax);
        }
        configurations[configurations_size] = tmp_conf;
        configurations_number.push_back(pose_number.at(ip));
        configurations_size++;
      }
    }
    ROS_WARN("%s <<<< Checking Collisions and Connect the %zu poses DT %fsec", hdr.c_str(), ik_res.solutions.size(), (ros::Time::now()-st).toSec());

    ROS_WARN("%s >>>> Connect To Approach", hdr.c_str());
    if ((inode < travel.size()-1) && enable_approach)
    {
      /* */ ROS_DEBUG("return to approach");

      if (checker->checkPath(last_node->getConfiguration(), approach))
      {
        last_q = approach;
        new_node = std::make_shared<pathplan::Node>(approach);
        pathplan::ConnectionPtr conn = std::make_shared<pathplan::Connection>(last_node, new_node);
        conn->add();
        conn->setCost(metrics->cost(last_node->getConfiguration(), approach));
        connections.push_back(conn);
        order_pose_number.push_back(-10);
        tree->addNode(new_node);
        last_node = new_node;
      }
      else
      {

        pathplan::SubtreePtr subtree=std::make_shared<pathplan::Subtree>(tree,last_node);

        Eigen::VectorXd actual_lb=lb;
        Eigen::VectorXd actual_ub=ub;
        for (unsigned int iax=0;iax<lb.size();iax++)
        {
          if (!std::isnan(elongation(iax)) && elongation(iax)>=0)
          {
            double min_value=std::min(last_node->getConfiguration()(iax),new_node->getConfiguration()(iax));
            double max_value=std::max(last_node->getConfiguration()(iax),new_node->getConfiguration()(iax));
            actual_lb(iax)=std::max(lb(iax),min_value-elongation(iax));
            actual_ub(iax)=std::min(ub(iax),max_value+elongation(iax));

          }
        }
        pathplan::InformedSamplerPtr sampler=std::make_shared<pathplan::InformedSampler>(last_node->getConfiguration(),
                                                                                         new_node->getConfiguration(),
                                                                                         actual_lb,
                                                                                         actual_ub);

        pathplan::NodePtr g = std::make_shared<pathplan::Node>(new_node->getConfiguration());
        pathplan::MultigoalSolver solver(metrics,checker,sampler);
        if(solver.config(solver_nh) && solver.addGoal(g))
        {
          pathplan::PathPtr solution;
          if (solver.solve(solution,1e5,max_computation_time))
          {
            last_q = approach;

            last_node = g;
            solution->setTree(tree);

            pathplan::PathLocalOptimizer path_opt(checker, metrics);
            path_opt.setPath(solution);
            path_opt.solve(solution,100000);
            std::vector<pathplan::ConnectionPtr> tmp_connections=solution->getConnections();

            for (size_t iconnection = 0; iconnection < tmp_connections.size(); iconnection++)
            {
              connections.push_back(tmp_connections.at(iconnection));
              order_pose_number.push_back(-10);
            }
          }
          else
          {
            ROS_WARN("Unable to find a solution to back to approach");
          }
        }
        else
        {
          ROS_WARN("The solver configuration failed, likely tyhe approach goal cannot be added...weird ...");
        }
      }
    }
    ROS_WARN("%s <<<< Connect To Approach DT %fsec", hdr.c_str(), (ros::Time::now()-st).toSec());

    ros::param::set("/complete/number_of_fail_poses", int(fail_poses.poses.size()));
    ros::param::set("/complete/number_of_no_feasible_ik_poses", int(no_feasible_ik_poses.poses.size()));
    ros::param::set("/complete/number_of_no_ik_poses", int(no_ik_poses.poses.size()));
    failed_poses_pub.publish(fail_poses);
    no_feasible_ik_poses_pub.publish(no_feasible_ik_poses);
    no_ik_poses_pub.publish(no_ik_poses);
  }
  ROS_INFO("[%s] complete the task", pnh.getNamespace().c_str());
  ROS_INFO("[%s] order_pose_number %zu", pnh.getNamespace().c_str(), order_pose_number.size());
  ROS_INFO("[%s] configurations_number %zu", pnh.getNamespace().c_str(), configurations_number.size());

  if (connections.size() > 0)
  {
    pathplan::Path path(connections, metrics, checker);
    XmlRpc::XmlRpcValue xml_path_original = path.toXmlRpcValue();
    ROS_WARN("Original || Size Cloud: %d", xml_path_original.size());
    ROS_WARN("Original || Size Ordered Pose Number: %zu", order_pose_number.size());
    ROS_WARN("Original || Size Cloud Connections: %zu", path.getConnections().size());
    if(1)
    {
      std::map<int,int> map_node_from_orig_to_new;
      pathplan::PathPtr resampled_path=path.resample(max_path_step, map_node_from_orig_to_new);
      assert(map_node_from_orig_to_new.size()==order_pose_number.size());

      std::vector<int> resampled_order_pose_number(resampled_path->getConnections().size()+1);
      for (size_t isampled=0;isampled<resampled_order_pose_number.size();isampled++)
      {
        auto it = map_node_from_orig_to_new.find(isampled);
        if( it != map_node_from_orig_to_new.end() )
        {
          resampled_order_pose_number.at(isampled) = order_pose_number.at(it->first);
        }
        else
        {
          resampled_order_pose_number.at(isampled) = -10;
        }
      }
      XmlRpc::XmlRpcValue xml_path = resampled_path->toXmlRpcValue();
      ROS_WARN("Resampled || Size Cloud: %d", xml_path.size());
      ROS_WARN("Resampled || Ordered Pose Number: %zu", resampled_order_pose_number.size());
      pnh.setParam("/complete/path/cloud", xml_path);
      pnh.setParam("/complete/path/cloud_pose_number", resampled_order_pose_number);
    }
    else
    {    
      XmlRpc::XmlRpcValue xml_path = path.toXmlRpcValue();
      pnh.setParam("/complete/path/cloud", xml_path);
      pnh.setParam("/complete/path/cloud_pose_number", order_pose_number);
    }
    pnh.setParam("/complete/configurations", configurations);
    pnh.setParam("/complete/configurations_number", configurations_number);


    res.success = true;
  }
  else
  {
    res.message = pnh.getNamespace() + ": Path has zero length, Connection size = 0";
    ROS_ERROR("%s", res.message.c_str());
    res.success = false;
    return true;
  }
  return true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test_subtree");
  ros::NodeHandle nh;

  ros::ServiceServer srv = nh.advertiseService("/compute_path", pathCb);
  ros::spin();
  return 0;
}
