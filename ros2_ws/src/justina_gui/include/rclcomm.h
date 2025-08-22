#ifndef RCLCOMM_H
#define RCLCOMM_H
#include <QObject>
#include <QThread>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"
class RclComm : public QThread//, rclcpp::Node
{
    Q_OBJECT
public:
    RclComm(); 
    //void recv_callback(const std_msgs::msg::String &msg);
    
    void start_publishing_cmd_vel(double linear_x, double linear_y, double angular);
    void start_publishing_cmd_vel(double linear_x, double angular);
    void stop_publishing_cmd_vel();

private:
    bool _publishing_cmd_vel;
    geometry_msgs::msg::Twist _cmd_vel;
    std::shared_ptr<rclcpp::Node> _node;
    rclcpp::TimerBase::SharedPtr _timer;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr _pub_cmd_vel;
signals:
    //void emitTopicData(QString);
};
#endif // RCLCOMM_H
