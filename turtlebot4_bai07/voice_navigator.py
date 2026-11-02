"""Phần 4: voice command (/voice/text) -> region -> Nav2 NavigateToPose.

    ros2 run turtlebot4_bai07 voice_navigator
    ros2 topic pub --once /voice/text std_msgs/msg/String "{data: 'đi đến phòng thí nghiệm'}"   # without a mic

Every sentence is parsed against regions.yaml (regions.py): "đi đến khu vực A" / "đi đến phòng thí
nghiệm" -> one goal; "từ khu vực A đến khu vực B" -> B if the robot is already in A, otherwise A
then B; "dừng lại" cancels. A place that is not on the map is reported and the robot does not move.
The goal of a region is its goal point (segment_map), heading along the leg so the robot does not
turn on the spot at the end; the robot pose comes from TF map -> base_link (AMCL).
A new command replaces the running one. The regions are drawn in RViz2 (/regions/markers) with
the current command and status (/voice_nav/status, also as text in RViz2).
"""
import math
import os
import unicodedata

import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from turtlebot4_bai07.regions import RegionMap


def ascii_text(text):
    """RViz2's marker font has no Vietnamese glyphs: 'Phòng thí nghiệm' -> 'Phong thi nghiem'."""
    t = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


class VoiceNavigator(Node):
    def __init__(self):
        super().__init__("voice_navigator")
        share = get_package_share_directory("turtlebot4_bai07")
        self.declare_parameter("regions_file", os.path.join(share, "maps", "regions.yaml"))
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("fill_step", 2)          # cells between region fill points in RViz2
        self.rmap = RegionMap(self.get_parameter("regions_file").value)
        self.robot_frame = self.get_parameter("robot_frame").value

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.region_pub = self.create_publisher(MarkerArray, "regions/markers", latched)
        self.info_pub = self.create_publisher(MarkerArray, "voice_nav/markers", latched)
        self.status_pub = self.create_publisher(String, "voice_nav/status", 10)
        self.create_subscription(String, "voice/text", self._on_text, 10)

        self.route, self.leg = [], 0          # regions to visit, index of the current one
        self.goal_handle, self.seq = None, 0  # seq: ignore results of replaced goals
        self.heard, self.status = "", "Waiting for a voice command"
        self.last_feedback = 0.0
        self._publish_regions()
        self._publish_info()
        names = ", ".join(r.title for r in self.rmap.regions)
        self.get_logger().info(f"{len(self.rmap.regions)} regions: {names}")
        self.get_logger().info("Waiting for voice commands on /voice/text")

    # ---------------------------------------------------------------- commands
    def _on_text(self, msg):
        self.heard = msg.data
        cmd = self.rmap.parse(msg.data)
        self.get_logger().info(f"heard \"{msg.data}\" -> {cmd['action']}")
        if cmd["action"] == "stop":
            self._cancel()
            self._set_status("Stopped (dừng lại)")
        elif cmd["action"] == "error":
            if cmd["reason"] == "unknown_place":
                self._set_status(f"No region \"{cmd['place']}\" on the map - robot does not move", warn=True)
            else:
                self._set_status(f"Not a command: \"{cmd['text']}\"", warn=True)
        else:
            self._start(cmd["route"], cmd["src"])

    def _start(self, route, src):
        here = self._robot_region()
        if src is not None and len(route) == 2:
            if here is src:
                route = route[1:]
                note = f"already in {src.label}, "
            else:
                note = f"in {here.label if here else '?'}, via {src.label}: "
        else:
            note = ""
        self._cancel(quiet=True)
        self.route, self.leg = route, 0
        self.get_logger().info(f"route: {note}{' -> '.join(r.title for r in route)}")
        self._send_leg()

    def _send_leg(self):
        region = self.route[self.leg]
        pose = self._robot_pose()
        gx, gy = region.goal["x"], region.goal["y"]
        yaw = math.atan2(gy - pose[1], gx - pose[0]) if pose else 0.0
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(gx), float(gy)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        if not self.nav.wait_for_server(timeout_sec=2.0):
            self._set_status("Nav2 (navigate_to_pose) is not running", warn=True)
            self.route = []
            return
        self.seq += 1
        seq = self.seq
        step = f" ({self.leg + 1}/{len(self.route)})" if len(self.route) > 1 else ""
        self._set_status(f"Going to {region.title}{step}")
        future = self.nav.send_goal_async(goal, feedback_callback=lambda fb: self._on_feedback(fb, seq))
        future.add_done_callback(lambda f: self._on_accepted(f, seq))

    def _on_accepted(self, future, seq):
        handle = future.result()
        if seq != self.seq:            # replaced meanwhile
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self._set_status(f"Nav2 rejected the goal of {self.route[self.leg].title}", warn=True)
            self.route = []
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, seq))

    def _on_feedback(self, feedback, seq):
        now = self.get_clock().now().nanoseconds * 1e-9
        if seq != self.seq or now - self.last_feedback < 3.0:
            return
        self.last_feedback = now
        fb = feedback.feedback
        region = self.route[self.leg]
        self.get_logger().info(f"  -> {region.title}: {fb.distance_remaining:.1f} m left, "
                               f"{fb.number_of_recoveries} recoveries")
        self._publish_info(extra=f"{fb.distance_remaining:.1f} m left")

    def _on_result(self, future, seq):
        if seq != self.seq:
            return
        status = future.result().status
        self.goal_handle = None
        region = self.route[self.leg]
        if status == GoalStatus.STATUS_SUCCEEDED:
            pose = self._robot_pose()
            inside = self.rmap.region_at(*pose[:2]) if pose else None
            check = "inside" if inside is region else f"robot in {inside.label if inside else '?'}"
            if self.leg + 1 < len(self.route):
                self.get_logger().info(f"reached {region.title} ({check}), next leg")
                self.leg += 1
                self._send_leg()
            else:
                self._set_status(f"Arrived at {region.title} ({check})")
                self.route = []
        elif status == GoalStatus.STATUS_CANCELED:
            self.route = []
        else:
            self._set_status(f"Nav2 could not reach {region.title} (status {status})", warn=True)
            self.route = []

    def _cancel(self, quiet=False):
        self.seq += 1
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            if not quiet:
                self.get_logger().info("goal cancelled")
        self.goal_handle, self.route = None, []

    # ---------------------------------------------------------------- robot
    def _robot_pose(self):
        try:
            t = self.tf.lookup_transform("map", self.robot_frame, Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def _robot_region(self):
        pose = self._robot_pose()
        return self.rmap.region_at(*pose[:2]) if pose else None

    # ---------------------------------------------------------------- display
    def _set_status(self, text, warn=False):
        self.status = text
        if warn:   # one logger call per severity: rclpy refuses two severities from one line
            self.get_logger().warning(text)
        else:
            self.get_logger().info(text)
        self.status_pub.publish(String(data=text))
        self._publish_info()

    def _publish_regions(self):
        arr = MarkerArray()
        step = max(1, int(self.get_parameter("fill_step").value))
        res = self.rmap.resolution
        for r in self.rmap.regions:
            color = ColorRGBA(r=r.color[0], g=r.color[1], b=r.color[2], a=0.35)
            fill = self._marker("region_fill", r.id, Marker.CUBE_LIST, color)
            fill.scale.x = fill.scale.y = step * res
            fill.scale.z = 0.01
            rows, cols = (self.rmap.labels == r.id).nonzero()
            for row, col in zip(rows, cols):
                if row % step == 0 and col % step == 0:
                    x = self.rmap.origin[0] + (col + step / 2) * res
                    y = self.rmap.origin[1] + (self.rmap.height - row - step / 2) * res
                    fill.points.append(Point(x=x, y=y, z=0.01))
            outline = self._marker("region_outline", r.id, Marker.LINE_STRIP,
                                   ColorRGBA(r=r.color[0] * 0.6, g=r.color[1] * 0.6, b=r.color[2] * 0.6, a=1.0))
            outline.scale.x = 0.05
            outline.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in r.polygon + r.polygon[:1]]
            gx, gy = r.goal["x"], r.goal["y"]
            goal = self._marker("region_goal", r.id, Marker.CYLINDER, ColorRGBA(r=0.85, g=0.1, b=0.1, a=0.9))
            goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = gx, gy, 0.02
            goal.scale.x = goal.scale.y = 0.2
            goal.scale.z = 0.04
            letter = self._marker("region_label", r.id, Marker.TEXT_VIEW_FACING, ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0))
            letter.pose.position.x, letter.pose.position.y, letter.pose.position.z = gx, gy + 0.75, 0.3
            letter.scale.z = 0.9
            letter.text = r.label
            name = self._marker("region_name", r.id, Marker.TEXT_VIEW_FACING, ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0))
            name.pose.position.x, name.pose.position.y, name.pose.position.z = gx, gy - 0.45, 0.3
            name.scale.z = 0.42
            name.text = ascii_text(r.name)
            arr.markers += [fill, outline, goal, letter, name]
        self.region_pub.publish(arr)

    def _publish_info(self, extra=""):
        arr = MarkerArray()
        text = self._marker("voice_status", 0, Marker.TEXT_VIEW_FACING, ColorRGBA(r=1.0, g=0.85, b=0.2, a=1.0))
        text.pose.position.x, text.pose.position.y, text.pose.position.z = 0.0, 11.0, 0.5
        text.scale.z = 0.6
        lines = [f"Heard: \"{ascii_text(self.heard)}\"" if self.heard else "Voice navigation",
                 ascii_text(self.status) + (f" - {extra}" if extra else "")]
        text.text = "\n".join(lines)
        target = self._marker("voice_target", 0, Marker.LINE_STRIP, ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0))
        target.scale.x = 0.14
        if self.route:
            r = self.route[self.leg]
            target.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in r.polygon + r.polygon[:1]]
        else:
            target.action = Marker.DELETE
        arr.markers += [text, target]
        self.info_pub.publish(arr)

    @staticmethod
    def _marker(ns, mid, mtype, color):
        m = Marker(ns=ns, id=mid, type=mtype, action=Marker.ADD, color=color)
        m.header.frame_id = "map"
        m.pose.orientation.w = 1.0
        return m


def main():
    rclpy.init()
    node = VoiceNavigator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node._cancel(quiet=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
