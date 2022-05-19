# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet

import pygame
import OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront

import random
import time
import numpy as np
import sys
from gym import spaces
import gym

import os
import math
import pybullet
import pybullet_data
from datetime import datetime
import pybullet_data
from collections import namedtuple

# from attrdict import AttrDict

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_with_camera.urdf"
TREE_URDF_PATH = "./ur_e_description/urdf/tree.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
CUBE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "cube_small.urdf")


# x,y,z distance
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# x,y distance
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)


class ur5GymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 # useIK=True,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=100,
                 # numControlledJoints=3, # XYZ, we use IK here!
                 simulatedGripper=False,
                 randObjPos=False,
                 task=0,  # here target number
                 learning_param=0,
                 width=424,
                 height=240):

        self.renders = renders
        self.actionRepeat = actionRepeat

        # setup pybullet sim:
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)

        pybullet.setTimeStep(1. / 240.)
        pybullet.setGravity(0, 0, -10)
        pybullet.setRealTimeSimulation(False)
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME,1)
        pybullet.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-30,
                                            cameraTargetPosition=[0, 0, 0])

        # setup robot arm:
        self.end_effector_index = 7
        self.table = pybullet.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
        self.tree= pybullet.loadURDF(TREE_URDF_PATH, [0., 0.0, 0.0], [0, 0, 0, 1])
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.ur5 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.num_joints = pybullet.getNumJoints(self.ur5)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                               "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo",
                                     ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                      "controllable"])
        self.proj_mat = pybullet.computeProjectionMatrixFOV(
            fov=42.0, aspect=width / height, nearVal=0.01,
            farVal=10.0)
        self.joints = dict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info

        # object:
        self.initial_obj_pos = [0.8, 0.1, 0.0]  # initial object pos
        self.obj = pybullet.loadURDF(CUBE_URDF_PATH, self.initial_obj_pos)

        self.name = 'ur5GymEnv'
        self.simulatedGripper = simulatedGripper
        self.action_dim = 4
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated = False
        self.randObjPos = randObjPos
        self.observation = np.array(0)

        self.task = task
        self.learning_param = learning_param

        self._action_bound = 1.0  # delta limits
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.reset()
        high = np.array([10] * self.observation.shape[0])
        self.observation_space = spaces.Box(-high, high, dtype='float32')
        self.scene = pywavefront.Wavefront('tree.obj', collect_faces=True)



    def getTreePoints(self, count):

        tree_oreint = pybullet.getQuaternionFromEuler([0,0,1.514])
        point=[]
        colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=.005)
        visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=.005,rgbaColor =[1,0,0,1])

        for i in range(count):

            scene_box = self.scene.vertices[i]

            tree_w_frame = pybullet.multiplyTransforms([-.5,0,-.5],tree_oreint,[scene_box[0]*.1,scene_box[1]*.1,scene_box[2]*.1],[0,0,0,1])
            position=[tree_w_frame[0][0],tree_w_frame[0][1],tree_w_frame[0][2],0]
            point.append(position)
            sphereUid = pybullet.createMultiBody(0.0, colSphereId, visualShapeId, [position[0],position[1],position[2]], [0,0,0,1])
        return point


    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.05] * len(poses),
            forces=forces
        )

    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1, 2, 3, 4, 5, 6])
        joints = [i[0] for i in j]
        return joints

    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            # print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False

    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        # print(quaternion)
        # quaternion = (0,1,0,1)
        lower_limits = [-math.pi] * 6
        upper_limits = [math.pi] * 6
        joint_ranges = [2 * math.pi] * 6
        # rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
        rest_poses = [(-0.34, -1.57, 1.80, -1.57, -1.57, 0.00)]  # rest pose of our ur5 robot

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion,
            jointDamping=[0.01] * 6, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges,
            restPoses=rest_poses
        )
        return joint_angles

    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

    def reset(self):
        self.stepCounter = 0
        self.terminated = False
        self.ur5_or = [0.0, 1 / 2 * math.pi, 0.0]

        # pybullet.addUserDebugText('X', self.obj_pos, [0,1,0], 1) # display goal
        if self.randObjPos:
            self.initial_obj_pos = [0.6 + (random.random() - 0.5) * 0.1, (random.random() - 0.5) * 0.5, 0.0]
            # print(self.initial_obj_pos)
        pybullet.resetBasePositionAndOrientation(self.obj, self.initial_obj_pos, [0., 0., 0., 1.0])  # reset object pos

        # reset robot simulation and position:
        joint_angles = (-0.34, -1.57, 1.80, -1.57, -1.57, 0.00)  # pi/2 = 1.5707
        self.set_joint_angles(joint_angles)

        # step simualator:
        for i in range(100):
            pybullet.stepSimulation()

        # get obs and return:
        self.getExtendedObservation()
        return self.observation

    def step(self, action):

        action = np.array(action)
        arm_action = action[0:self.action_dim - 1].astype(float)  # dX, dY, dZ - range: [-1,1]
        print("arm action", arm_action)
        gripper_action = action[self.action_dim - 1].astype(float)  # gripper - range: [-1=closed,1=open]

        # get current position:
        cur_p = self.get_current_pose()
        print("ppppppppp",cur_p)
        # add delta position:
        new_p = np.array(cur_p[0]) + arm_action
        # actuate:
        joint_angles = self.calculate_ik(new_p, self.ur5_or)  # XYZ and angles set to zero
        self.set_joint_angles(joint_angles)
        self.set_camera(cur_p[0], cur_p[1])
        # step simualator:
        for i in range(self.actionRepeat):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1. / 240.)

        self.getExtendedObservation()
        reward = self.compute_reward(self.achieved_goal, self.desired_goal, None)
        done = self.my_task_done()

        info = {'is_success': False}
        if self.terminated == self.task:
            info['is_success'] = True

        self.stepCounter += 1

        return self.observation, reward, done, info

    # observations are: arm (tip/tool) position, arm acceleration, ...
    def getExtendedObservation(self):
        # sensor values:
        # js = self.get_joint_angles()

        tool_pos = self.get_current_pose()[0]  # XYZ, no angles
        self.obj_pos, _ = pybullet.getBasePositionAndOrientation(self.obj)
        objects_pos = self.obj_pos
        goal_pos = self.obj_pos

        self.observation = np.array(np.concatenate((tool_pos, objects_pos)))
        self.achieved_goal = np.array(np.concatenate((objects_pos, tool_pos)))
        self.desired_goal = np.array(goal_pos)

    def my_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        c = (self.terminated == True or self.stepCounter > self.maxSteps)
        return c

    def set_camera(self, pose, orientation):
        ##!!!!!!!!!!!!!!! Need to rewrite this
        camMat = pybullet.getMatrixFromQuaternion(orientation)
        # upVector = [0,0,1]
        forwardVec = [camMat[0], camMat[3], camMat[6]]
        # sideVec =  [camMat[1],camMat[4],camMat[7]]
        camUpVec = [camMat[2], camMat[5], camMat[8]]
        camTarget = [pose[0] + forwardVec[0] * 10, pose[1] + forwardVec[1] * 10, pose[2] + forwardVec[2] * 10]
        camUpTarget = [pose[0] + camUpVec[0], pose[1] + camUpVec[1], pose[2] + camUpVec[2]]
        viewMat = pybullet.computeViewMatrix(pose, camTarget, camUpVec)
        pybullet.getCameraImage(320, 200, viewMatrix=viewMat, projectionMatrix=self.proj_mat,
                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = np.zeros(1)

        grip_pos = achieved_goal[-3:]

        self.target_dist = goal_distance(grip_pos, desired_goal)
        # print(grip_pos, desired_goal, self.target_dist)

        # check approach velocity:
        # tv = self.tool.getVelocity()
        # approach_velocity = np.sum(tv)

        # print(approach_velocity)
        # input()

        reward += -self.target_dist * 10

        # task 0: reach object:
        if self.target_dist < self.learning_param:  # and approach_velocity < 0.05:
            self.terminated = True
            # print('Successful!')

        # penalize if it tries to go lower than desk / platform collision:
        # if grip_trans[1] < self.desired_goal[1]-0.08: # lower than position of object!
        # reward[i] += -1
        # print('Penalty: lower than desk!')

        # check collisions:
        if self.check_collisions():
            reward += -1
            # print('Collision!')

        # print(target_dist, reward)
        # input()

        return reward
