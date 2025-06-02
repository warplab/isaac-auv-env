Code for Learning to Swim: Reinforcement Learning for 6-DOF Control of Thruster-driven Autonomous Underwater Vehicles.

![Overview](./imgs/qual-overview.png)

Links: [arxiv paper](https://arxiv.org/abs/2410.00120)

To install, requires IsaacSim v4.0.0 and IsaacLab v1.0.0:
- Install IsaacSim v4.0.0 (https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)
  - Download and unzip the archived binaries into a new folder (i.e. "IsaacSim")
  - Soft link Isaac lab and Isaac sim 
    ```
    cd <IsaacLab_Path> 
    ln -s <IsaacSim_Path> _isaac_sim
    ```
   
- Install IsaacLab v1.0.0 (https://isaac-sim.github.io/IsaacLab/v1.0.0/source/setup/installation/index.html)
  ```
  git clone --branch v1.0.0 https://github.com/isaac-sim/IsaacLab.git
  ```

- Add Required patch (https://github.com/isaac-sim/IsaacLab/pull/1808/files/8af43cb048cdaa976c24a0f2b569ea9e45db533d)
  - For v1.0.0 IsaacLab, patch needs to be applied to <IsaacLab_Path>/source/extensions/omni.isaac.lab_tasks/setup.py before running install script

- Clone this repository:

  - If using docker container:
  ```
  cd <IsaacLab_Path>/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-warpauv-env
  git clone https://github.com/warplab/isaac-auv-env.git
  ```

  - If using workstation install:
  ```
  git clone https://github.com/warplab/isaac-auv-env.git
  cd <IsaacLab_Path>/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/
  ln -s <isaac-auv-env_Path> isaac-auv-env
  ```
  (Note: if using a workstation install, you can follow the docker instructions as well, but the soft link seems cleaner for local development. Docker is painful when working with links)
  
- Update pip and Install with:
  ```
  <IsaacSim_Path>/kit/python/bin/python3 -m pip install --upgrade pip
  ./isaaclab.sh -i
  ```
 

To run training:
```
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-WarpAUV-Direct-v1 --num_envs 2048
```

Additional notes:

 - To import a URDF file into USD format for IsaacLab, you can first export a ROS xacro file into URDF, and then import that URDF file into the IsaacSim URDFImporter Workflow.

 ```
 rosrun xacro xacro --inorder -o <output.urdf> <input.xacro>
 ./isaaclab.sh -p source/standalone/tools/convert_urdf.py  <input_urdf> <output_usd> --merge-joints --make-instance
 ```

 - Generally converges in about 400 iterations with 2048 environments and achieves mean total reward ~95-100. Lowering action penalty often helps if there are issues with convergence.

To cite:
```
@inproceedings{caiLearningSwimReinforcement2025,
  title = {Learning to {{Swim}}: {{Reinforcement Learning}} for 6-{{DOF Control}} of {{Thruster-driven Autonomous Underwater Vehicles}}},
  booktitle = {2025 {{IEEE International Conference}} on {{Robotics}} and {{Automation}} ({{ICRA}})},
  author = {Cai, Levi and Chang, Kevin and Girdhar, Yogesh},
  date = {2025},
  url = {https://arxiv.org/abs/2410.00120},
  eventtitle = {2025 {{IEEE International Conference}} on {{Robotics}} and {{Automation}} ({{ICRA}})}
}

```
