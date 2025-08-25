#include "rclcomm.h"
#include <string>
#include <chrono>

RclComm::RclComm(): Node("justina_gui_node")
{
    std::cout << "Initializing RclComm node ..." << std::endl;
    this->_node = rclcpp::Node::make_shared("ros2_qt_demo"); 
    this->_pub_cmd_vel =  this->_node->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    // auto timer_callback =
    // 	[this]() -> void {
    // 	    std::cout << "Testing callback" << std::endl;
    // 	};
    //_timer = this->create_wall_timer(std::chrono::milliseconds(500), std::bind(&RclComm::timer_callback, this));
    //this->_subp = this->_node->create_subscription<std_msgs::msg::String>("chat_qt", 10, std::bind(&RclComm::recv_callback, this, std::placeholders::_1));
    this->_sub_test = this->_node->create_subscription<std_msgs::msg::Float32>("test", 10, std::bind(&RclComm::callback_current_arm_pose, this, std::placeholders::_1));
}

void RclComm::timer_callback()
{
    std::cout << "Testing callback" << std::endl;
}

void RclComm::callback_current_arm_pose(const std_msgs::msg::Float32 &msg)
{
    std::cout << "Received float " << msg.data << std::endl;
}

void RclComm::publish_cmd_vel(double linear_x, double linear_y, double angular)
{
    geometry_msgs::msg::Twist msg;
    msg.linear.x = linear_x;
    msg.linear.y = linear_y;
    msg.angular.z =  angular;
    this->_pub_cmd_vel->publish(msg);
}

void RclComm::publish_cmd_vel(double linear_x, double angular)
{
    this->publish_cmd_vel(linear_x, 0, angular);
}

void RclComm::start_publishing_cmd_vel(double linear_x, double linear_y, double angular)
{
    _cmd_vel.linear.x = linear_x;
    _cmd_vel.linear.y = linear_y;
    _cmd_vel.angular.z = angular;
    _publishing_cmd_vel = true;
}

void RclComm::start_publishing_cmd_vel(double linear_x, double angular)
{
    this->start_publishing_cmd_vel(linear_x, 0, angular);
}

void RclComm::stop_publishing_cmd_vel()
{
    _cmd_vel.linear.x = 0;
    _cmd_vel.linear.y = 0;
    _cmd_vel.angular.z = 0;
    _publishing_cmd_vel = false;
}

// void RclComm::recv_callback(const std_msgs::msg::String &msg)
// {
//     emit emitTopicData("pub send a msgs:" + QString::fromStdString(msg.data));
// }

// // spin
// void RclComm::sping()
// {
//     rclcpp::spin_some(_node);
// }
