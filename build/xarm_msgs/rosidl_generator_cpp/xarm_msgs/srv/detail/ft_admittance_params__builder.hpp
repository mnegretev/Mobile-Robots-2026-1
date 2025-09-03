// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from xarm_msgs:srv/FtAdmittanceParams.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "xarm_msgs/srv/ft_admittance_params.hpp"


#ifndef XARM_MSGS__SRV__DETAIL__FT_ADMITTANCE_PARAMS__BUILDER_HPP_
#define XARM_MSGS__SRV__DETAIL__FT_ADMITTANCE_PARAMS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "xarm_msgs/srv/detail/ft_admittance_params__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace xarm_msgs
{

namespace srv
{

namespace builder
{

class Init_FtAdmittanceParams_Request_b
{
public:
  explicit Init_FtAdmittanceParams_Request_b(::xarm_msgs::srv::FtAdmittanceParams_Request & msg)
  : msg_(msg)
  {}
  ::xarm_msgs::srv::FtAdmittanceParams_Request b(::xarm_msgs::srv::FtAdmittanceParams_Request::_b_type arg)
  {
    msg_.b = std::move(arg);
    return std::move(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Request msg_;
};

class Init_FtAdmittanceParams_Request_k
{
public:
  explicit Init_FtAdmittanceParams_Request_k(::xarm_msgs::srv::FtAdmittanceParams_Request & msg)
  : msg_(msg)
  {}
  Init_FtAdmittanceParams_Request_b k(::xarm_msgs::srv::FtAdmittanceParams_Request::_k_type arg)
  {
    msg_.k = std::move(arg);
    return Init_FtAdmittanceParams_Request_b(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Request msg_;
};

class Init_FtAdmittanceParams_Request_m
{
public:
  explicit Init_FtAdmittanceParams_Request_m(::xarm_msgs::srv::FtAdmittanceParams_Request & msg)
  : msg_(msg)
  {}
  Init_FtAdmittanceParams_Request_k m(::xarm_msgs::srv::FtAdmittanceParams_Request::_m_type arg)
  {
    msg_.m = std::move(arg);
    return Init_FtAdmittanceParams_Request_k(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Request msg_;
};

class Init_FtAdmittanceParams_Request_c_axis
{
public:
  explicit Init_FtAdmittanceParams_Request_c_axis(::xarm_msgs::srv::FtAdmittanceParams_Request & msg)
  : msg_(msg)
  {}
  Init_FtAdmittanceParams_Request_m c_axis(::xarm_msgs::srv::FtAdmittanceParams_Request::_c_axis_type arg)
  {
    msg_.c_axis = std::move(arg);
    return Init_FtAdmittanceParams_Request_m(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Request msg_;
};

class Init_FtAdmittanceParams_Request_coord
{
public:
  Init_FtAdmittanceParams_Request_coord()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FtAdmittanceParams_Request_c_axis coord(::xarm_msgs::srv::FtAdmittanceParams_Request::_coord_type arg)
  {
    msg_.coord = std::move(arg);
    return Init_FtAdmittanceParams_Request_c_axis(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::xarm_msgs::srv::FtAdmittanceParams_Request>()
{
  return xarm_msgs::srv::builder::Init_FtAdmittanceParams_Request_coord();
}

}  // namespace xarm_msgs


namespace xarm_msgs
{

namespace srv
{

namespace builder
{

class Init_FtAdmittanceParams_Response_message
{
public:
  explicit Init_FtAdmittanceParams_Response_message(::xarm_msgs::srv::FtAdmittanceParams_Response & msg)
  : msg_(msg)
  {}
  ::xarm_msgs::srv::FtAdmittanceParams_Response message(::xarm_msgs::srv::FtAdmittanceParams_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Response msg_;
};

class Init_FtAdmittanceParams_Response_ret
{
public:
  Init_FtAdmittanceParams_Response_ret()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FtAdmittanceParams_Response_message ret(::xarm_msgs::srv::FtAdmittanceParams_Response::_ret_type arg)
  {
    msg_.ret = std::move(arg);
    return Init_FtAdmittanceParams_Response_message(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::xarm_msgs::srv::FtAdmittanceParams_Response>()
{
  return xarm_msgs::srv::builder::Init_FtAdmittanceParams_Response_ret();
}

}  // namespace xarm_msgs


namespace xarm_msgs
{

namespace srv
{

namespace builder
{

class Init_FtAdmittanceParams_Event_response
{
public:
  explicit Init_FtAdmittanceParams_Event_response(::xarm_msgs::srv::FtAdmittanceParams_Event & msg)
  : msg_(msg)
  {}
  ::xarm_msgs::srv::FtAdmittanceParams_Event response(::xarm_msgs::srv::FtAdmittanceParams_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Event msg_;
};

class Init_FtAdmittanceParams_Event_request
{
public:
  explicit Init_FtAdmittanceParams_Event_request(::xarm_msgs::srv::FtAdmittanceParams_Event & msg)
  : msg_(msg)
  {}
  Init_FtAdmittanceParams_Event_response request(::xarm_msgs::srv::FtAdmittanceParams_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_FtAdmittanceParams_Event_response(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Event msg_;
};

class Init_FtAdmittanceParams_Event_info
{
public:
  Init_FtAdmittanceParams_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FtAdmittanceParams_Event_request info(::xarm_msgs::srv::FtAdmittanceParams_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_FtAdmittanceParams_Event_request(msg_);
  }

private:
  ::xarm_msgs::srv::FtAdmittanceParams_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::xarm_msgs::srv::FtAdmittanceParams_Event>()
{
  return xarm_msgs::srv::builder::Init_FtAdmittanceParams_Event_info();
}

}  // namespace xarm_msgs

#endif  // XARM_MSGS__SRV__DETAIL__FT_ADMITTANCE_PARAMS__BUILDER_HPP_
