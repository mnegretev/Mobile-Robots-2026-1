from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ros_gz_sim_pkg_path  = get_package_share_directory('ros_gz_sim')
    gazebo_envs_pkg_path = get_package_share_directory('gazebo_envs')
    gz_lauch_path        = PathJoinSubstitution([ros_gz_sim_pkg_path, 'launch', 'gz_sim.launch.py'])
    world_file_subpath   = 'worlds/simple_house.world'

    return LaunchDescription([
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            PathJoinSubstitution([gazebo_envs_pkg_path, 'models'])
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_lauch_path),
            launch_arguments={
                'gz_args':[PathJoinSubstitution([gazebo_envs_pkg_path, world_file_subpath])],
                'on_exit_shutdown':'True'
            }.items(),
        ),
    ])
