#!/usr/bin/env python3
"""Bài 07 bring-up: TurtleBot 4 in the bài 04 maze with a few obstacles that are not on the map,
the bài 04 map with AMCL + Nav2, RViz2 (map, segmented regions, Nav2 path), voice_navigator and
speech_recognition_node (microphone).

The simulation itself is turtlebot4_bai04/sim.launch.py (Gazebo headless on macOS, bridges,
Create 3 nodes, AMCL/Nav2 parameter files).

    ros2 launch turtlebot4_bai07 bai07.launch.py                 # everything, microphone
    ros2 launch turtlebot4_bai07 bai07.launch.py mic:=false      # commands from /voice/wav or /voice/text
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription,
                            SetLaunchConfiguration, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg = get_package_share_directory("turtlebot4_bai07")
    pkg_bai04 = get_package_share_directory("turtlebot4_bai04")
    pkg_tb4_nav = get_package_share_directory("turtlebot4_navigation")

    args = [
        DeclareLaunchArgument("world", default_value="maze_obstacles",
                              description="maze_obstacles (bài 04 maze + obstacles) or maze"),
        DeclareLaunchArgument("map", default_value=os.path.join(pkg, "maps", "map.yaml")),
        DeclareLaunchArgument("regions", default_value=os.path.join(pkg, "maps", "regions.yaml"),
                              description="Regions written by segment_map"),
        DeclareLaunchArgument("rviz", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("mic", default_value="true", choices=["true", "false"],
                              description="false: speech_recognition_node only recognizes /voice/wav files"),
        DeclareLaunchArgument("device", default_value="", description="Microphone (sounddevice index or name)"),
        DeclareLaunchArgument("grammar", default_value="true", choices=["true", "false"],
                              description="Limit Vosk to the command sentences"),
    ]

    # Gazebo looks for maze_obstacles.sdf here.
    world_path = [AppendEnvironmentVariable(name, os.path.join(pkg, "worlds"))
                  for name in ("IGN_GAZEBO_RESOURCE_PATH", "GZ_SIM_RESOURCE_PATH")]

    # The include sets rviz:=false for the whole launch (no scoping in Humble): keep ours first.
    keep_rviz = SetLaunchConfiguration("bai07_rviz", LaunchConfiguration("rviz"))
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_bai04, "launch", "sim.launch.py")),
        launch_arguments={"world": LaunchConfiguration("world"), "localization": "true", "nav2": "false",
                          "map": LaunchConfiguration("map"), "rviz": "false"}.items(),
    )
    # Nav2 with the bài 04 parameters, except that the planner may not cross unknown cells
    # (the unexplored parts of the map are walls or closed areas).
    nav2 = TimerAction(period=12.0, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_tb4_nav, "launch", "nav2.launch.py")),
        launch_arguments={"namespace": "", "use_sim_time": "true",
                          "params_file": RewrittenYaml(source_file=os.path.join(pkg_bai04, "config", "nav2.yaml"),
                                                       param_rewrites={"allow_unknown": "false"},
                                                       convert_types=True)}.items(),
    )])

    # On macOS the RViz2 OpenGL renderer can segfault while the simulation starts: respawn it.
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", os.path.join(pkg, "rviz", "bai07.rviz"), "-f", "map"],
        parameters=[{"use_sim_time": True}],
        respawn=True, respawn_delay=2.0,
        condition=IfCondition(LaunchConfiguration("bai07_rviz")),
    )

    navigator = Node(
        package="turtlebot4_bai07", executable="voice_navigator", name="voice_navigator", output="screen",
        parameters=[{"use_sim_time": True, "regions_file": LaunchConfiguration("regions")}],
    )
    speech = Node(
        package="turtlebot4_bai07", executable="speech_recognition_node", name="speech_recognition_node",
        output="screen",
        parameters=[{
            "regions_file": LaunchConfiguration("regions"),
            "source": PythonExpression(["'mic' if '", LaunchConfiguration("mic"), "' == 'true' else 'none'"]),
            "device": LaunchConfiguration("device"),
            "grammar": ParameterValue(LaunchConfiguration("grammar"), value_type=bool),
        }],
    )

    # Vosk and the microphone start once Gazebo, AMCL and Nav2 are up.
    return LaunchDescription(args + world_path + [keep_rviz, sim, nav2, rviz, navigator,
                                                  TimerAction(period=20.0, actions=[speech])])
