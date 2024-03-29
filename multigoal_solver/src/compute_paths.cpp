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
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <ik_solver_msgs/GetIkArray.h>
#include <std_srvs/Trigger.h>
#include <moveit_msgs/GetPlanningScene.h>



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
  if (!pnh.getParam("group_name",group_name))
  {
    ROS_ERROR("%s/group_name is not defined",pnh.getNamespace().c_str());
    res.success=false;
    return true;
  }

  ros::ServiceClient ps_client = nh.serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");
  ps_client.waitForExistence();
  moveit_msgs::GetPlanningScene ps_srv;
  if (!ps_client.call(ps_srv))
  {
    ROS_ERROR("Error on  get_planning_scene srv not ok");
    res.success=false;
    return true;
  }

  std::vector<double> w;
  if (!pnh.getParam("weight",w))
  {
    ROS_ERROR("%s/weight is not defined",pnh.getNamespace().c_str());
    res.success=false;
    return true;
  }

  moveit::planning_interface::MoveGroupInterface move_group(group_name);
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr           kinematic_model = robot_model_loader.getModel();
  planning_scene::PlanningScenePtr     planning_scene = std::make_shared<planning_scene::PlanningScene>(kinematic_model);
  std::vector<std::string> joint_names = kinematic_model->getJointModelGroup(group_name)->getActiveJointModelNames();
  unsigned int dof = joint_names.size();




  int num_threads =pnh.param("number_of_threads",5);
  double steps=pnh.param("collision_steps",0.01);
  double maximum_distance=pnh.param("maximum_distance",0.01);

  pathplan::CollisionCheckerPtr checker = std::make_shared<pathplan::MoveitCollisionChecker>(planning_scene, group_name,maximum_distance);
  checker->setPlanningSceneMsg(ps_srv.response.scene);
  pathplan::MetricsPtr metrics=std::make_shared<pathplan::Metrics>();

  XmlRpc::XmlRpcValue travel;

  if (!pnh.getParam("travel",travel))
  {
    ROS_ERROR("%s/travel is not defined",pnh.getNamespace().c_str());
    res.success=false;
    return true;
  }

  sensor_msgs::PointCloud2Ptr pc;

  ros::Subscriber pc_sub=nh.subscribe<sensor_msgs::PointCloud2>("/point_cloud2",100,boost::bind(pointCloudCb,_1,boost::ref(pc)));

  std::string tool_name;
  ros::ServiceClient ik_client;
  if (!nh.getParam("/tool_name",tool_name))
    ik_client=nh.serviceClient<ik_solver_msgs::GetIkArray>("/ik_solver/get_ik_array");
  else
    ik_client=nh.serviceClient<ik_solver_msgs::GetIkArray>("/"+tool_name+"_ik_solver/get_ik_array");

  ros::Publisher failed_poses_pub=nh.advertise<geometry_msgs::PoseArray>("fail_poses",10,true);


  ROS_INFO("%s is waiting for the point cloud",pnh.getNamespace().c_str());

  ros::Rate lp(50);
  while (ros::ok())
  {
    ros::spinOnce();
    if (pc)
      break;
    lp.sleep();

  }

  int data_size=pc->data.size()/(sizeof(float));


  std::vector<float> data(data_size);
  memcpy(&(data.at(0)),&(pc->data.at(0)),pc->data.size());


  int n_points=data_size/pc->fields.size();
  geometry_msgs::PoseArray all_poses;
  all_poses.header.frame_id=pc->header.frame_id;
  all_poses.poses.resize(n_points);
  std::vector<int> group(n_points);

  int idx=0;
  for (int ip=0;ip<n_points;ip++)
  {
    geometry_msgs::Pose& p=all_poses.poses.at(ip);

    p.position.x=data.at(idx++);
    p.position.y=data.at(idx++);
    p.position.z=data.at(idx++);
    p.orientation.x=data.at(idx++);
    p.orientation.y=data.at(idx++);
    p.orientation.z=data.at(idx++);
    p.orientation.w=data.at(idx++);
    group.at(ip)=data.at(idx++);
  }



  Eigen::VectorXd q;
  Eigen::VectorXd last_q;
  geometry_msgs::PoseArray fail_poses;
  fail_poses.header.frame_id=pc->header.frame_id;

  bool first_node=true;
  bool first_time=true;
  pathplan::TreePtr tree;

  pathplan::NodePtr last_node;
  pathplan::NodePtr new_node;

  std::vector<pathplan::ConnectionPtr> connections;
  std::vector<int> order_pose_number;

  XmlRpc::XmlRpcValue configurations;
  int configurations_size=0;

  for (int inode=0;inode<travel.size();inode++)
  {
    std::string node=travel[inode]["node"];
    int ik_sol=(int)static_cast<double>(travel[inode]["ik"]);

    if (!nh.getParam("/goals/"+node+"/joint_names",joint_names))
    {
      ROS_WARN_STREAM("unable to read parameter /goals/"<< node+"/joint_names");
      //res.success=false;
      //return true;
      continue;


    }

    ik_solver_msgs::GetIkArrayRequest ik_req;
    ik_solver_msgs::GetIkArrayResponse ik_res;
    ik_req.poses.header=all_poses.header;
    ik_req.seed_joint_names=joint_names;
    ik_solver_msgs::Configuration seed;



    std::string s=node;
    s.erase(std::remove_if(std::begin(s), std::end(s), [](char ch) { return !std::isdigit(ch); }), s.end());
    int igroup=std::stoi(s);
    ROS_INFO("node %s ik=%d, group=%d",node.c_str(),ik_sol,igroup);

    std::string tree_name="/goals/"+node+"/iksol"+std::to_string(ik_sol);
    std::vector<double> iksol;



    if (!nh.getParam(tree_name+"/root",iksol))
    {
      ROS_ERROR_STREAM("unable to read parameter "<< tree_name+"/root");
      res.success=false;
      return true;
    }

    seed.configuration=iksol;
    ik_req.seeds.push_back(seed);

    q=Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(iksol.data(), iksol.size());
    Eigen::VectorXd approach=q;
    Eigen::MatrixXd weight(q.size(),q.size());
    weight.setIdentity();
    for (int iax=0;iax<q.size();iax++)
      weight(iax,iax)=w.at(iax);



    if (first_node)
    {
      last_q=q;
      pathplan::NodePtr root = std::make_shared<pathplan::Node>(q);
      last_node=root;

      if (!checker->check(q))
      {
        ROS_FATAL("root not is unreachable: this should not happen");
        res.success=false;
        return true;
      }

      tree=std::make_shared<pathplan::Tree>(root,maximum_distance,checker,metrics);
      first_node=false;
    }
    else
    {
      if (checker->checkPath(last_node->getConfiguration(),approach))
      {
        last_q=approach;
        new_node=std::make_shared<pathplan::Node>(approach);
        pathplan::ConnectionPtr conn=std::make_shared<pathplan::Connection>(last_node,new_node);
        conn->add();
        connections.push_back(conn);
        order_pose_number.push_back(-10);
        tree->addNode(new_node);
        last_node=new_node;
        ROS_INFO("connect with next keypoint");
      }
    }



    std::vector<int> pose_number;
    for (int ip=0;ip<n_points;ip++)
    {
      if (group.at(ip)==igroup)
      {
        ik_req.poses.poses.push_back(all_poses.poses.at(ip));
        pose_number.push_back(ip);
      }
    }

    if (!ik_client.call(ik_req,ik_res))
    {
      ROS_ERROR("%s unable to call ik service",pnh.getNamespace().c_str());
      res.success=false;
      return true;
    }


    ROS_INFO("processing %zu poses",ik_res.solutions.size());



    for (size_t ip=0;ip<ik_res.solutions.size();ip++)
    {
      ROS_DEBUG("Pose %zu of %zu (keypoint %s)",ip,ik_res.solutions.size(),node.c_str());
      ik_solver_msgs::IkSolution& ik= ik_res.solutions.at(ip);
      bool connected=false;

      std::multimap<double,Eigen::VectorXd> ordered_configurations;
      for (ik_solver_msgs::Configuration& c: ik.configurations)
      {
        q=Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(c.configuration.data(), c.configuration.size());
        //double dist=(q-last_q).norm();
        if (checker->check(q))
        {
          double dist=std::sqrt(q.transpose()*weight*q);
          ordered_configurations.insert(std::pair<double,Eigen::VectorXd>(dist,q));
        }
      }

      if (!first_time)
      {
        for (const std::pair<double,Eigen::VectorXd>& p: ordered_configurations)
        {
          if (checker->checkPath(last_node->getConfiguration(),p.second))
          {
            last_q=p.second;
            connected=true;

            new_node=std::make_shared<pathplan::Node>(p.second);
            pathplan::ConnectionPtr conn=std::make_shared<pathplan::Connection>(last_node,new_node);
            conn->add();
            connections.push_back(conn);
            order_pose_number.push_back(pose_number.at(ip));
            tree->addNode(new_node);
            last_node=new_node;
            break;
          }
        }
      }
      if (!connected)
      {
        ROS_DEBUG("Pose %zu of %zu (keypoint %s): Try connect",ip,ik_res.solutions.size(),node.c_str());
        tree->changeRoot(last_node);

        for (const std::pair<double,Eigen::VectorXd>& p: ordered_configurations)
        {
          if (tree->connect(p.second,new_node))
          {
            last_q=p.second;
            connected=true;

            first_time=false;
            last_node=new_node;
            std::vector<pathplan::ConnectionPtr> tmp_connections=tree->getConnectionToNode(new_node);
            for (size_t iconnection=0;iconnection<tmp_connections.size();iconnection++)
            {
              connections.push_back(tmp_connections.at(iconnection));
              order_pose_number.push_back(pose_number.at(ip));
            }

            break;
          }
        }
      }



      if (!connected)
      {
        ROS_INFO("Unreachable: Pose %zu of %zu (keypoint %s)",ip,ik_res.solutions.size(),node.c_str());

        fail_poses.poses.push_back(ik_req.poses.poses.at(ip));
      }
      else
      {
        XmlRpc::XmlRpcValue tmp_conf;
        for (int iax=0;iax<last_q.size();iax++)
        {
          tmp_conf[iax]=last_q(iax);
        }
        configurations[configurations_size]=tmp_conf;
        configurations_size++;
      }
    }



    ROS_INFO("return to approach");
    if (checker->checkPath(last_node->getConfiguration(),approach))
    {
      last_q=approach;
      new_node=std::make_shared<pathplan::Node>(approach);
      pathplan::ConnectionPtr conn=std::make_shared<pathplan::Connection>(last_node,new_node);
      conn->add();
      connections.push_back(conn);
      order_pose_number.push_back(-10);
      tree->addNode(new_node);
      last_node=new_node;
    }
    else
    {
      tree->changeRoot(last_node);

      if (tree->connect(approach,new_node))
      {
        last_q=approach;

        last_node=new_node;
        std::vector<pathplan::ConnectionPtr> tmp_connections=tree->getConnectionToNode(new_node);
        for (size_t iconnection=0;iconnection<tmp_connections.size();iconnection++)
        {
          connections.push_back(tmp_connections.at(iconnection));
          order_pose_number.push_back(-10);
        }
      }
      else
        ROS_ERROR("Unable to come back to approach");


    }

    failed_poses_pub.publish(fail_poses);

  }
  ROS_INFO("%s complete the task",pnh.getNamespace().c_str());

  if (connections.size()>0)
  {
    pathplan::Path path(connections,metrics,checker);
    XmlRpc::XmlRpcValue xml_path=path.toXmlRpcValue();
    pnh.setParam("/complete/path/cloud",xml_path);
    pnh.setParam("/complete/path/cloud_pose_number",order_pose_number);
    pnh.setParam("/complete/configurations",configurations);
  }
  res.success=true;
  return true;
}



int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_subtree");
  ros::NodeHandle nh;

  ros::ServiceServer srv=nh.advertiseService("/compute_path",pathCb);
  ros::spin();
  return 0;
}
