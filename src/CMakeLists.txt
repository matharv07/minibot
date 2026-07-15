cmake_minimum_required(VERSION 3.8)
project(minibot)

# Install all resource directories so that ament_index can find them
# and $(find minibot) resolves at runtime.
find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    config
    launch
    meshes
    urdf
    pacmanbot
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY
    scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/omni_teleop.py
    scripts/sonar_to_laserscan.py
    scripts/scan_merger.py
    scripts/maze_generator.py
    scripts/nrf24_bridge.py
    scripts/spawn_arena.py
    scripts/pacman_node.py
    scripts/ghost_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
