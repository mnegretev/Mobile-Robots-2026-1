from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import xacro
import os

def generate_launch_description():
    ros_gz_sim_pkg_path  = get_package_share_directory('ros_gz_sim')
    gazebo_envs_pkg_path = get_package_share_directory('gazebo_envs')
    gz_sim_launch_path   = PathJoinSubstitution([ros_gz_sim_pkg_path, 'launch', 'gz_sim.launch.py'])
    gz_spawn_launch_path = PathJoinSubstitution([ros_gz_sim_pkg_path, 'launch', 'gz_spawn_model.launch.py'])
    world_file_path      = PathJoinSubstitution([gazebo_envs_pkg_path, 'worlds', 'simple_house.world'])

    description_pkg_path  = get_package_share_directory('justina_description')
    xacro_file_path = os.path.join(description_pkg_path, 'urdf','justina.xacro')
    robot_description_content = xacro.process_file(xacro_file_path).toxml()
    xarm_descrip_pkg_path = get_package_share_directory('xarm_description')
    
    robot_state_publisher_params = [{'robot_description': robot_description_content}]
    gz_bridge_params_path = os.path.join(gazebo_envs_pkg_path, 'config', 'gz_bridge.yaml')
    rviz_config_file = os.path.join(gazebo_envs_pkg_path, 'config', 'simple_house.rviz')

    return LaunchDescription([
        # SetEnvironmentVariable(
        #     'GZ_SIM_RESOURCE_PATH',
        #     PathJoinSubstitution([gazebo_envs_pkg_path, 'models'])
        # ),
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(gazebo_envs_pkg_path, 'models') + ":" + os.path.join(xarm_descrip_pkg_path, "..")
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_sim_launch_path),
            launch_arguments={
                'gz_args':[f'-r ', world_file_path],
                'on_exit_shutdown':'True',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_spawn_launch_path),
            launch_arguments={
                'world':'default',
                'topic':'/robot_description',
                'entity_name': 'justina',
                'x': '-4.0',
                'y': '0.0',
                'z': '1.0',
                'Y': '1.5708'
            }.items(),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=robot_state_publisher_params
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '--ros-args', '-p',
                f'config_file:={gz_bridge_params_path}'
            ],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
        )
    ])
