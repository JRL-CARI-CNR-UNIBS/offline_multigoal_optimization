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

void pointCloudCb(const sensor_msgs::PointCloud2ConstPtr& msg, sensor_msgs::PointCloud2Ptr& pc)
{
  pc.reset(new sensor_msgs::PointCloud2(*msg));
}


//bool computePathFromHere(const std::vector<ik_solver_msgs::IkSolution>& iksols,
//                         const Eigen::VectorXd& q,
//                         const size_t& idx,
//                         pathplan::CollisionCheckerPtr& checker,
//                         std::vector<Eigen::VectorXd>& nodes)
//{

//  if (idx>=iksols.size())
//    return true;

//  const ik_solver_msgs::IkSolution& iksol=iksols.at(idx);

//  std::multimap<double,Eigen::VectorXd> ordered_configurations;

//  for (const ik_solver_msgs::Configuration& c: iksol.configurations)
//  {
//    Eigen::VectorXd child=Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(c.configuration.data(), c.configuration.size());
//    double dist=(child-q).norm();
//    ordered_configurations.insert(std::pair<double,Eigen::VectorXd>(dist,child));
//  }

//  for (const std::pair<double,Eigen::VectorXd>& p: ordered_configurations)
//  {
//    if (checker->checkPath(q,p.second))
//    {
//      std::vector<Eigen::VectorXd> child_nodes;
//      if (computePathFromHere(iksols,
//                              p.second,
//                              idx+1,
//                              checker,
//                              child_nodes))
//      {
//        nodes.clear();
//        nodes.push_back(q);
//        nodes.insert(nodes.end(), child_nodes.begin(), child_nodes.end());

//        ROS_INFO("lista: ");
//        for (const Eigen::VectorXd& x: nodes)
//        {
//          ROS_INFO_STREAM("- "<<x.transpose());
//        }
//        return true;
//      }
//    }
//  }
//  return false;
//}


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

  std::vector<double> w;
  if (!pnh.getParam("weight",w))
  {
    ROS_ERROR("%s/weight is not defined",pnh.getNamespace().c_str());
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




  int num_threads =pnh.param("number_of_threads",5);
  double steps=pnh.param("collision_steps",0.01);
  double maximum_distance=pnh.param("maximum_distance",0.01);

  pathplan::CollisionCheckerPtr checker = std::make_shared<pathplan::ParallelMoveitCollisionChecker>(planning_scene, group_name,num_threads,steps);
  pathplan::MetricsPtr metrics=std::make_shared<pathplan::Metrics>();

  XmlRpc::XmlRpcValue travel;

  if (!pnh.getParam("travel",travel))
  {
    ROS_ERROR("%s/travel is not defined",pnh.getNamespace().c_str());
    return 0;
  }

  sensor_msgs::PointCloud2Ptr pc;

  ros::Subscriber pc_sub=nh.subscribe<sensor_msgs::PointCloud2>("/point_cloud2",100,boost::bind(pointCloudCb,_1,boost::ref(pc)));

  ros::ServiceClient ik_client=nh.serviceClient<ik_solver_msgs::GetIkArray>("/ik_solver/get_ik_array");

  ros::Publisher poses_pub=nh.advertise<geometry_msgs::PoseArray>("req_poses",10,true);


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

  for (int inode=0;inode<travel.size();inode++)
  {
    std::string node=travel[inode]["node"];
    int ik_sol=(int)static_cast<double>(travel[inode]["ik"]);


    std::string s=node;
    s.erase(std::remove_if(std::begin(s), std::end(s), [](char ch) { return !std::isdigit(ch); }), s.end());
    int igroup=std::stoi(s);
    ROS_INFO("node %s ik=%d, group=%d",node.c_str(),ik_sol,igroup);


    std::string tree_name="/goals/"+node+"/iksol"+std::to_string(ik_sol);
    std::vector<double> iksol;

    if (!nh.getParam(tree_name+"/root",iksol))
    {
      ROS_ERROR_STREAM("unable to read parameter "<< tree_name+"/root");
      return 0;
    }

    q=Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(iksol.data(), iksol.size());
    last_q=q;
    pathplan::NodePtr root = std::make_shared<pathplan::Node>(q);

    Eigen::MatrixXd weight(q.size(),q.size());
    weight.setIdentity();
    for (int iax=0;iax<q.size();iax++)
      weight(iax,iax)=w.at(iax);

    if (!checker->check(q))
    {
      ROS_FATAL("this should not happen");
      return 0;
    }

    pathplan::TreePtr tree=std::make_shared<pathplan::Tree>(root,maximum_distance,checker,metrics);



    ik_solver_msgs::GetIkArrayRequest req;
    ik_solver_msgs::GetIkArrayResponse res;
    req.poses.header=all_poses.header;
    ik_solver_msgs::Configuration seed;
    seed.configuration.push_back(ik_sol);
    req.seeds.push_back(seed);

    for (int ip=0;ip<n_points;ip++)
    {
      if (group.at(ip)==igroup)
      {
        req.poses.poses.push_back(all_poses.poses.at(ip));
      }
    }

    ROS_INFO("%zu poses",req.poses.poses.size());



    if (!ik_client.call(req,res))
    {
      ROS_ERROR("%s unable to call ik service",pnh.getNamespace().c_str());
      return 0;
    }


    pathplan::NodePtr new_node;
    pathplan::NodePtr last_node;



    ROS_INFO("processing %zu poses",res.solutions.size());

    std::vector<pathplan::ConnectionPtr> connections;

    bool first_time=true;
    for (size_t ip=0;ip<res.solutions.size();ip++)
    {
      ik_solver_msgs::IkSolution& ik= res.solutions.at(ip);
      bool connected=false;

      std::multimap<double,Eigen::VectorXd> ordered_configurations;
      for (ik_solver_msgs::Configuration& c: ik.configurations)
      {
        q=Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(c.configuration.data(), c.configuration.size());
        //double dist=(q-last_q).norm();
        double dist=std::sqrt(q.transpose()*weight*q);
        ordered_configurations.insert(std::pair<double,Eigen::VectorXd>(dist,q));
      }

      if (first_time)
      {
        for (const std::pair<double,Eigen::VectorXd>& p: ordered_configurations)
        {
          if (tree->connect(p.second,new_node))
          {
            last_q=p.second;
            connected=true;


            first_time=false;
            last_node=new_node;
            connections=tree->getConnectionToNode(new_node);


            break;
          }
        }
      }
      else
      {
        for (const std::pair<double,Eigen::VectorXd>& p: ordered_configurations)
        {
          if (checker->checkPath(last_node->getConfiguration(),p.second))
          {
            last_q=p.second;
            connected=true;

            new_node=std::make_shared<pathplan::Node>(p.second);
            pathplan::ConnectionPtr conn=std::make_shared<pathplan::Connection>(last_node,new_node);
            connections.push_back(conn);
            tree->addNode(new_node);
            last_node=new_node;
            break;
          }
        }
      }

      if (!connected)
        fail_poses.poses.push_back(req.poses.poses.at(ip));

    }


    poses_pub.publish(fail_poses);
    pnh.setParam("/tmp_tree",tree->toXmlRpcValue());

    ROS_INFO("SAVING PATH %s connections=%zu",tree_name.c_str(),connections.size());
    if (connections.size()>0)
    {
      pathplan::Path path(connections,metrics,checker);
      XmlRpc::XmlRpcValue xml_path=path.toXmlRpcValue();
      pnh.setParam(tree_name+"/path/cloud",xml_path);
    }
    else
    {
      pnh.deleteParam(tree_name+"/path/cloud");
    }

  }
  ROS_INFO("%s complete the task",pnh.getNamespace().c_str());
  ros::spin();
  return 0;
}
