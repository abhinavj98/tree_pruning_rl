# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet



# import pygame
# import OpenGL
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLU import *
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
#from attrdict import AttrDict
from enum import Enum

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_with_camera.urdf"
TREE_URDF_PATH = "./ur_e_description/urdf/tree.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")

# x,y,z distance
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def goal_reward(current, previous, target):
    #assert goal_a.shape == goal_b.shape
    #assert goal_a.shape == target.shape
    diff_prev = goal_distance(previous, target)
    diff_curr = goal_distance(current, target)
    reward = diff_prev - diff_curr
    return reward

# x,y distance
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)

class action(Enum):
    up = 1
    down = 2
    left = 3
    right = 4
    forward = 5
    backward = 6
    roll_up = 7
    roll_down = 8
    pitch_up = 9
    pitch_down = 10
    yaw_up = 11
    yaw_down = 12

class ur5GymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 # useIK=True,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=100,
                 # numControlledJoints=3, # XYZ, we use IK here!
                 simulatedGripper=False,
                 randObjPos=True,
                 task=0, # here target number
                 learning_param=0,
                 width = 424,
                 height = 240):

        self.renders = renders
        self.actionRepeat = actionRepeat

        # setup pybullet sim:
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)

        pybullet.setTimeStep(1./240.)
        pybullet.setGravity(0,0,-10)
        pybullet.setRealTimeSimulation(False)
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME,1)
        pybullet.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=73.95, cameraPitch=-38.48, cameraTargetPosition=[1.04,-0.06,0.14])
        self.sphereUid = -1
        # setup robot arm:
        self.end_effector_index = 7
        #self.table = pybullet.loadURDF(TABLE_URDF_PATH, [0.2, 0, -0.6300], [0, 0, 0, 1])
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.ur5 = pybullet.loadURDF(ROBOT_URDF_PATH, [0.8, 0, 0], [0, 0, 0, 1], flags=flags)
        self.num_joints = pybullet.getNumJoints(self.ur5)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
        self.proj_mat = pybullet.computeProjectionMatrixFOV(
            fov=42, aspect = width / height, nearVal=0.01,
            farVal=10.0)
        self.joints = dict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            print(i, jointName)
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info

        # object:
        self.initial_obj_pos = [0.8, 0.1, 0.0] # initial object pos

        self.name = 'ur5GymEnv'
        self.simulatedGripper = simulatedGripper
        # discrete action
        self.action_dim = 4
        #self.action_dim = 12
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated = False
        self.randObjPos = randObjPos
        self.observation = np.array(0)
        self.previous_pose = (np.array(0), np.array(0))

        self.task = task
        self.learning_param = learning_param
        # discrete action
        # self._action_bound = 1.0 # delta limits
        # action_high = np.array([self._action_bound] * self.action_dim)
        # self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.action_space = spaces.Discrete(4)
        self.actions = {'up':1,
                        'down':2,
                        'left' : 3,
                        'right' : 4,
                        'forward' : 5,
                        'backward' : 6,
                        'roll_up' : 7,
                        'roll_down': 8,
                        'pitch_up' : 9,
                        'pitch_down' : 10,
                        'yaw_up' : 11,
                        'yaw_down' : 12}

        self.rev_actions = {v: k for k,v in self.actions.items()}

        self.tree = pybullet.loadURDF(TREE_URDF_PATH, [2.3, 0.0, 0.0], [0, 0, 0, 1], globalScaling=1)
        self.scene = pywavefront.Wavefront('project_tree.obj', collect_faces=True)
        self.tree_reachble = []
        self.tree_target=self.getTreePoints(len(self.scene.vertices))
        print(self.tree_target)
        self.reset()
        high = np.array([10]*self.observation.shape[0])
        self.observation_space = spaces.Box(-high, high, dtype='float32')

    def getTreePoints(self, count):

        
        point=[]
       
        ur5_base_pos,_ = pybullet.getBasePositionAndOrientation(self.ur5)
        tree_pos, tree_orient = pybullet.getBasePositionAndOrientation(self.tree)
        tree_orient = pybullet.getQuaternionFromEuler([0,0,1.54])
        for i in range(count):

            scene_box = self.scene.vertices[i]
            tree_w_frame = pybullet.multiplyTransforms(tree_pos,tree_orient,[scene_box[0]*.1,scene_box[1]*.1,scene_box[2]*.1],[0,0,0,1])
            position=[tree_w_frame[0][0]-0.7,tree_w_frame[0][1],tree_w_frame[0][2]-.5]
            point.append(position)
           
            dist=np.sqrt((np.square(ur5_base_pos[0]-position[0]))+((np.square(ur5_base_pos[1]-position[1]))+((np.square(ur5_base_pos[2]-position[2])))))
            if dist <= 1. and position[2]>0.2:
                self.tree_reachble.append(position)


        print("got tree points")
        return self.tree_reachble




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
            targetVelocities=[0]*len(poses),
            positionGains=[0.05]*len(poses),
            forces=forces
        )

    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints


    def check_collisions(self):
        collisions = pybullet.getContactPoints(bodyA = self.ur5, bodyB = self.tree, linkIndexA=6)
        # print(collisions)
        for i in range(len(collisions)):
            if collisions[i][-6] < -0.001 :
                #print("[Collision detected!] {}, {}".format(datetime.now(), collisions[i]))
                return True
        return False


    def calculate_ik(self, position, orientation):
        #discrete actions
        # quaternion = pybullet.getQuaternionFromEuler(orientation)
        quaternion = orientation

        # quaternion = (0,1,0,1)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        # rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
        rest_poses = [(3.14,-1.57,1.80,0,0,0)]#[(-0.34, -1.57, 1.80, -1.57, -1.57, 0.00)] # rest pose of our ur5 robot

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion,
            jointDamping=[0.01]*6, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges,
            restPoses=rest_poses
        )
        return joint_angles


    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

    def set_camera(self, pose, orientation):
        ##!!!!!!!!!!!!!!! Need to rewrite this
        pose, orientation = self.get_current_pose()
        rot_mat = np.array(pybullet.getMatrixFromQuaternion(orientation)).reshape(3,3)
		#upVector = [0,0,1]
        #Initial vectors
        init_camera_vector = np.array([1,0, 0])#
        init_up_vector = np.array([0, 0,1]) #
        #Rotated vectors
        camera_vector = rot_mat.dot(init_camera_vector)
        up_vector = rot_mat.dot(init_up_vector)
        view_matrix = pybullet.computeViewMatrix(pose, pose + 0.1 * camera_vector, up_vector)
       
        return pybullet.getCameraImage(224, 224, viewMatrix = view_matrix, projectionMatrix = self.proj_mat, renderer = pybullet.ER_BULLET_HARDWARE_OPENGL)
        camMat = pybullet.getMatrixFromQuaternion(orientation)
        forwardVec = [camMat[0],camMat[3],camMat[6]]
		#sideVec =  [camMat[1],camMat[4],camMat[7]]
        camUpVec =  [camMat[2],camMat[5],camMat[8]]
        camTarget = [pose[0]+forwardVec[0]*10,pose[1]+forwardVec[1]*10,pose[2]+forwardVec[2]*10]
        camUpTarget = [pose[0]+camUpVec[0],pose[1]+camUpVec[1],pose[2]+camUpVec[2]]
        viewMat = pybullet.computeViewMatrix(pose, camTarget, camUpVec)
        return pybullet.getCameraImage(224, 224, viewMatrix = viewMat, projectionMatrix = self.proj_mat, renderer = pybullet.ER_BULLET_HARDWARE_OPENGL)

    @staticmethod
    def seperate_rgbd_rgb_d(rgbd, h = 224, w = 224):
        rgb = rgbd[2][:,:,0:3].reshape(3,h,w)/255
        depth = rgbd[3]
        return rgb, depth

    def reset(self):
        self.stepCounter = 0
        self.terminated = False
        self.ur5_or = [0.0, 1/2*math.pi, 0.0]
        cur_p = self.get_current_pose()
        rgbd = self.set_camera(cur_p[0], cur_p[1])
        self.rgb,  self.depth = self.seperate_rgbd_rgb_d(rgbd)
        # pybullet.addUserDebugText('X', self.obj_pos, [0,1,0], 1) # display goal
        if self.randObjPos:
            self.initial_obj_pos = random.sample(self.tree_target,1)[0]
        pybullet.removeBody(self.sphereUid)
        colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=.0002)
        visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=.02,rgbaColor =[1,0,0,1])
        self.sphereUid = pybullet.createMultiBody(0.0, colSphereId, visualShapeId, [self.initial_obj_pos[0],self.initial_obj_pos[1],self.initial_obj_pos[2]], [0,0,0,1])
        #print(self.initial_obj_pos)
        #print(self.tree_target)
        #pybullet.resetBasePositionAndOrientation(self.obj, self.tree_target[1], [0.,0.,0.,1.0]) # reset object pos
        #pybullet.resetBasePositionAndOrientation(self.obj, self.initial_obj_pos, [0.,0.,0.,1.0]) # reset object pos
        #colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=.005*3)
        #visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=.005*3, rgbaColor=[1, 0, 0, 1])
        #sphereUid = pybullet.createMultiBody(0.0, colSphereId, visualShapeId, [self.initial_obj_pos[0], self.initial_obj_pos[1], self.initial_obj_pos[2]], [0, 0, 0, 1])

        # reset robot simulation and position:
        # joint_angles = (-0.34, -1.57, 1.80, -1.57, -1.57, 0.00) # pi/2 = 1.5707
        joint_angles = (0, -1.57,1.80,-3.14,-1.57, -1.57)
        #joint_angles = (1.57,-1.57,1.80,-3.14,3.14,0)
        
        self.set_joint_angles(joint_angles)

        # step simualator:
        for i in range(100):
            pybullet.stepSimulation()

        # get obs and return:
        self.getExtendedObservation()
        return self.observation


    def step(self, action, debug = False):
        #discrete action
        deltaPose = np.array([0, 0, 0])
        deltaOrient= np.array([0, 0, 0])
        angle_scale = 1
        step_size =  0.5

        if action == self.actions['up']:
            deltaPose = [step_size, 0, 0,]

        if action == self.actions['down']:
            deltaPose = [-step_size, 0, 0]

        if action == self.actions['left']:
            deltaPose = [0, step_size, 0]

        if action == self.actions['right']:
            deltaPose = [0, -step_size, 0]

        if action == self.actions['forward']:
            deltaPose = [0, 0, step_size]

        if action == self.actions['backward']:
            deltaPose = [0, 0, -step_size]

        if action == self.actions['roll_up']:
            deltaOrient= [ step_size / angle_scale, 0, 0]

        if action == self.actions['roll_down']:
            deltaOrient= [ -step_size / angle_scale, 0, 0]

        if action == self.actions['pitch_up']:
            deltaOrient= [0, step_size / angle_scale, 0]

        if action == self.actions['pitch_down']:
            deltaOrient= [0, -step_size / angle_scale, 0]

        if action == self.actions['yaw_up']:
            deltaOrient= [0, 0, step_size / angle_scale]

        if action == self.actions['yaw_down']:
            deltaOrient= [0, 0, -step_size / angle_scale]
        # action = np.array(action)
        # arm_action = action[0:self.action_dim-1].astype(float) # dX, dY, dZ - range: [-1,1]
        # gripper_action = action[self.action_dim-1].astype(float) # gripper - range: [-1=closed,1=open]

        # get current position:
        cur_p = self.get_current_pose()
        self.previous_pose = cur_p
        # add delta position:
        # new_p = np.array(cur_p[0]) + arm_action
        new_position = np.array(cur_p[0]) + deltaPose
        new_orientation=np.array(cur_p[1]) + pybullet.getQuaternionFromEuler(deltaOrient)
        # actuate:

        joint_angles = self.calculate_ik(new_position, new_orientation) # XYZ and angles set to zero
        self.set_joint_angles(joint_angles)
        cur_p = self.get_current_pose()
        rgbd = self.set_camera(cur_p[0], cur_p[1])
        self.rgb,  self.depth = self.seperate_rgbd_rgb_d(rgbd)
        
        # step simualator:
        for i in range(self.actionRepeat):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)

        self.getExtendedObservation()
        reward = self.compute_reward(self.achieved_goal, self.achieved_orient, self.desired_goal, self.previous_goal, None)
        done = self.my_task_done()

        info = {'is_success': False}
        if self.terminated == self.task:
            info['is_success'] = True

        self.stepCounter += 1
        debug_img = None
        if debug:
            cam_prop =(1024, 768, (0.9961947202682495, -0.043577890843153, 0.07547912001609802, 0.0, 0.087155781686306, 0.49809736013412476, -0.8627299666404724, 0.0, -0.0, 0.8660255074501038, 0.5, 0.0, -1.0308130979537964, -0.04603677988052368, -1.7002619504928589, 1.0), (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (-0.07547912001609802, 0.8627299666404724, -0.5), (26565.193359375, 2324.154052734375, -0.0), (-871.5578002929688, 9961.947265625, 17320.5078125), 5.0, -30.0, 1.5, (1.0399999618530273, -0.05999999865889549, 0.14000000059604645))
            debug_img_rgbd = pybullet.getCameraImage(cam_prop[0], cam_prop[1], viewMatrix = cam_prop[2], projectionMatrix = cam_prop[3], renderer = pybullet.ER_BULLET_HARDWARE_OPENGL)
            #print(debug_img_rgbd[2][:,:,0:3])
            debug_img = debug_img_rgbd[2][:,:,0:3].reshape(cam_prop[1], cam_prop[0], 3).astype('uint8')
            
            #debug_img,  _ = self.seperate_rgbd_rgb_d(debug_img_rgbd, cam_prop[0], cam_prop[1])
        return self.observation, reward, done, debug_img, info


    # observations are: arm (tip/tool) position, arm acceleration, ...
    def getExtendedObservation(self):
        # sensor values:
        # js = self.get_joint_angles()

        tool_pos, tool_orient = self.get_current_pose()# XYZ, no angles
        objects_pos = self.initial_obj_pos
        goal_pos = self.initial_obj_pos

        self.observation = np.array(np.concatenate((tool_pos, objects_pos)))
        self.achieved_goal = np.array(tool_pos)
        self.desired_goal = np.array(goal_pos)
        self.previous_goal = np.array(self.previous_pose[0])
        self.previous_orient = np.array(self.previous_pose[1])
        self.achieved_orient=tool_orient


    def my_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        c = (self.terminated == True or self.stepCounter > self.maxSteps)
        return c


    def compute_reward(self, achieved_goal, achieved_orient, desired_goal, achieved_previous_goal, info):
        reward = np.zeros(1).astype('float32')
        #[roll_a, pitch_a, yaw_a] = pybullet.getEulerFromQuaternion(achieved_orient)

        self.target_reward = goal_reward(achieved_goal, achieved_previous_goal, desired_goal)
        self.target_dist = goal_distance(achieved_goal, desired_goal)
        # print(grip_pos, desired_goal, self.target_dist)


        # check approach velocity:
        # tv = self.tool.getVelocity()
        # approach_velocity = np.sum(tv)

        # print(approach_velocity)
        # input()
        reward += self.target_reward*10 #Mean around 0 -> Change in distance
        dist_reward = self.target_reward*10
        # task 0: reach object:
        terminate_reward = 0
        if self.target_dist < self.learning_param:  # and approach_velocity < 0.05:
            self.terminated = True
            terminate_reward = 1.5
            reward += 1.5
            print('Successful!')

        # check collisions:
        collision = False
        if self.check_collisions():
            reward += -0
            collision = True
            #print('Collision!')
        reward+= -0.05
        # print(target_dist, reward)
        # input()

        return (reward, dist_reward, terminate_reward, collision)

