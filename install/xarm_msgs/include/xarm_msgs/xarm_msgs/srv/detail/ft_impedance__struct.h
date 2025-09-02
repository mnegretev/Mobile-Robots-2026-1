// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from xarm_msgs:srv/FtImpedance.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "xarm_msgs/srv/ft_impedance.h"


#ifndef XARM_MSGS__SRV__DETAIL__FT_IMPEDANCE__STRUCT_H_
#define XARM_MSGS__SRV__DETAIL__FT_IMPEDANCE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'c_axis'
// Member 'm'
// Member 'k'
// Member 'b'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/FtImpedance in the package xarm_msgs.
typedef struct xarm_msgs__srv__FtImpedance_Request
{
  ///   - set_impedance
  ///   - set_impedance_config
  /// task frame (0: base frame. 1: tool frame)
  int16_t coord;
  /// a 6d vector of 0s and 1s. 1 means that robot will be admittance in the corresponding axis of the task frame.
  rosidl_runtime_c__int16__Sequence c_axis;
  ///   - set_impedance
  ///   - set_impedance_mbk
  /// 6d vector, mass. (kg)
  rosidl_runtime_c__float__Sequence m;
  /// 6d vector, stiffness coefficient
  rosidl_runtime_c__float__Sequence k;
  /// 6d vector, damping coefficient
  rosidl_runtime_c__float__Sequence b;
} xarm_msgs__srv__FtImpedance_Request;

// Struct for a sequence of xarm_msgs__srv__FtImpedance_Request.
typedef struct xarm_msgs__srv__FtImpedance_Request__Sequence
{
  xarm_msgs__srv__FtImpedance_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} xarm_msgs__srv__FtImpedance_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/FtImpedance in the package xarm_msgs.
typedef struct xarm_msgs__srv__FtImpedance_Response
{
  int16_t ret;
  rosidl_runtime_c__String message;
} xarm_msgs__srv__FtImpedance_Response;

// Struct for a sequence of xarm_msgs__srv__FtImpedance_Response.
typedef struct xarm_msgs__srv__FtImpedance_Response__Sequence
{
  xarm_msgs__srv__FtImpedance_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} xarm_msgs__srv__FtImpedance_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  xarm_msgs__srv__FtImpedance_Event__request__MAX_SIZE = 1
};
// response
enum
{
  xarm_msgs__srv__FtImpedance_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/FtImpedance in the package xarm_msgs.
typedef struct xarm_msgs__srv__FtImpedance_Event
{
  service_msgs__msg__ServiceEventInfo info;
  xarm_msgs__srv__FtImpedance_Request__Sequence request;
  xarm_msgs__srv__FtImpedance_Response__Sequence response;
} xarm_msgs__srv__FtImpedance_Event;

// Struct for a sequence of xarm_msgs__srv__FtImpedance_Event.
typedef struct xarm_msgs__srv__FtImpedance_Event__Sequence
{
  xarm_msgs__srv__FtImpedance_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} xarm_msgs__srv__FtImpedance_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // XARM_MSGS__SRV__DETAIL__FT_IMPEDANCE__STRUCT_H_
