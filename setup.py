from glob import glob

from setuptools import setup

package_name = "turtlebot4_bai07"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/maps", glob("maps/*")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Khanhkhan1",
    maintainer_email="khanh20022810@gmail.com",
    description="Bài 07: map segmentation into labelled regions + voice navigation with Vosk and Nav2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "segment_map = turtlebot4_bai07.map_segmentation:main",
            "speech_recognition_node = turtlebot4_bai07.speech_recognition_node:main",
            "voice_navigator = turtlebot4_bai07.voice_navigator:main",
        ],
    },
)
