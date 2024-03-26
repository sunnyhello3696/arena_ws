/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#ifdef COMPILE_WITH_ROS

#include "interface/IOROS.h"
#include "interface/KeyBoard.h"
#include <iostream>
#include <unistd.h>
#include <csignal>

void RosShutDown(int sig){
	ROS_INFO("ROS interface shutting down!");
	ros::shutdown();
}

IOROS::IOROS(bool blocking):IOInterface(){

    _nm = ros::NodeHandle("~");

    ROS_INFO("The control interface for ROS Gazebo simulation");
    
    params = new ROSParams();
    _nm.param<std::string>("robot_namespace", params->robotNamespace, "/UNKNOWN_NAMESPACE");
    _nm.param<int>("target_state", params->targetState, -1);

    // start subscriber
    initRecv(blocking);
    ros::AsyncSpinner subSpinner(1); // one threads
    subSpinner.start();
    usleep(300000);     //wait for subscribers start
    // initialize publisher
    initSend();   

    signal(SIGINT, RosShutDown);

    cmdPanel = new KeyBoard();
}

IOROS::~IOROS(){
    delete cmdPanel;
    delete params;
    ros::shutdown();
}

void IOROS::sendRecv(const LowlevelCmd *cmd, LowlevelState *state){
    sendCmd(cmd);
    recvState(state);

    state->userCmd = cmdPanel->getUserCmd();
    state->userValue = cmdPanel->getUserValue();
}

void IOROS::sendCmd(const LowlevelCmd *lowCmd){

    if(isPaused){
        return;
    }
    // lastPublish = clock;

    for(int i(0); i < 12; ++i){
        _lowCmd.motorCmd[i].mode = lowCmd->motorCmd[i].mode;
        _lowCmd.motorCmd[i].q = lowCmd->motorCmd[i].q;
        _lowCmd.motorCmd[i].dq = lowCmd->motorCmd[i].dq;
        _lowCmd.motorCmd[i].tau = lowCmd->motorCmd[i].tau;
        _lowCmd.motorCmd[i].Kd = lowCmd->motorCmd[i].Kd;
        _lowCmd.motorCmd[i].Kp = lowCmd->motorCmd[i].Kp;
    }
    for(int m(0); m < 12; ++m){
        if(!(
            std::isnan(_lowCmd.motorCmd[m].q)   ||
            std::isnan(_lowCmd.motorCmd[m].dq)  ||
            std::isnan(_lowCmd.motorCmd[m].tau) ||
            std::isnan(_lowCmd.motorCmd[m].Kd)  ||
            std::isnan(_lowCmd.motorCmd[m].Kp) 
        )){
            _servo_pub[m].publish(_lowCmd.motorCmd[m]);
        }
    }
    ros::spinOnce();
}

void IOROS::recvState(LowlevelState *state){

    for(int i(0); i < 12; ++i){
        state->motorState[i].q = _lowState.motorState[i].q;
        state->motorState[i].dq = _lowState.motorState[i].dq;
        state->motorState[i].ddq = _lowState.motorState[i].ddq;
        state->motorState[i].tauEst = _lowState.motorState[i].tauEst;
    }
    for(int i(0); i < 3; ++i){
        state->imu.quaternion[i] = _lowState.imu.quaternion[i];
        state->imu.accelerometer[i] = _lowState.imu.accelerometer[i];
        state->imu.gyroscope[i] = _lowState.imu.gyroscope[i];
    }
    state->imu.quaternion[3] = _lowState.imu.quaternion[3];
}

void IOROS::initSend(){
    ROS_INFO("advertising %s/XX_YY_controller/command", params->robotNamespace.c_str());
    _servo_pub[0] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/FR_hip_controller/command", 1);
    _servo_pub[1] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/FR_thigh_controller/command", 1);
    _servo_pub[2] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/FR_calf_controller/command", 1);
    _servo_pub[3] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/FL_hip_controller/command", 1);
    _servo_pub[4] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/FL_thigh_controller/command", 1);
    _servo_pub[5] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/FL_calf_controller/command", 1);
    _servo_pub[6] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/RR_hip_controller/command", 1);
    _servo_pub[7] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/RR_thigh_controller/command", 1);
    _servo_pub[8] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/RR_calf_controller/command", 1);
    _servo_pub[9] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace +  "/RL_hip_controller/command", 1);
    _servo_pub[10] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace + "/RL_thigh_controller/command", 1);
    _servo_pub[11] = _nm.advertise<unitree_legged_msgs::MotorCmd>(params->robotNamespace + "/RL_calf_controller/command", 1);
}

void IOROS::initRecv(bool blocking = false){

    if(blocking){

        ROS_INFO("waiting for clock");
        boost::shared_ptr<rosgraph_msgs::Clock const> msg;
        
        do{
            msg = ros::topic::waitForMessage<rosgraph_msgs::Clock>("/clock", _nm);
        }
        while(!msg);

        ROS_INFO("valid clock message received");
    }

    _clockSub = _nm.subscribe("/clock", 1, &IOROS::UpdateClock, this);

    std::string trunk_topic = params->robotNamespace + "/trunk_imu";
    ROS_INFO("subscribing %s", trunk_topic.c_str());
    _imu_sub = _nm.subscribe(trunk_topic, 1, &IOROS::imuCallback, this);

    ROS_INFO("subscribing %s/XX_YY_controller/state", params->robotNamespace.c_str());
    _servo_sub[0] = _nm.subscribe(params->robotNamespace +  "/FR_hip_controller/state", 1, &IOROS::FRhipCallback, this);
    _servo_sub[1] = _nm.subscribe(params->robotNamespace +  "/FR_thigh_controller/state", 1, &IOROS::FRthighCallback, this);
    _servo_sub[2] = _nm.subscribe(params->robotNamespace +  "/FR_calf_controller/state", 1, &IOROS::FRcalfCallback, this);
    _servo_sub[3] = _nm.subscribe(params->robotNamespace +  "/FL_hip_controller/state", 1, &IOROS::FLhipCallback, this);
    _servo_sub[4] = _nm.subscribe(params->robotNamespace +  "/FL_thigh_controller/state", 1, &IOROS::FLthighCallback, this);
    _servo_sub[5] = _nm.subscribe(params->robotNamespace +  "/FL_calf_controller/state", 1, &IOROS::FLcalfCallback, this);
    _servo_sub[6] = _nm.subscribe(params->robotNamespace +  "/RR_hip_controller/state", 1, &IOROS::RRhipCallback, this);
    _servo_sub[7] = _nm.subscribe(params->robotNamespace +  "/RR_thigh_controller/state", 1, &IOROS::RRthighCallback, this);
    _servo_sub[8] = _nm.subscribe(params->robotNamespace +  "/RR_calf_controller/state", 1, &IOROS::RRcalfCallback, this);
    _servo_sub[9] = _nm.subscribe(params->robotNamespace +  "/RL_hip_controller/state", 1, &IOROS::RLhipCallback, this);
    _servo_sub[10] = _nm.subscribe(params->robotNamespace + "/RL_thigh_controller/state", 1, &IOROS::RLthighCallback, this);
    _servo_sub[11] = _nm.subscribe(params->robotNamespace + "/RL_calf_controller/state", 1, &IOROS::RLcalfCallback, this);
}

void IOROS::UpdateClock(const rosgraph_msgs::Clock & msg){
    isPaused = clock == msg.clock;
    wasPaused |= isPaused;
    clock = msg.clock;

    if(wasPaused != isPaused){
        // std::cout << "was paused, but not anymore" << std::endl;
        // initRecv(true);
        wasPaused = false;
    }
}

void IOROS::imuCallback(const sensor_msgs::Imu & msg)
{ 
    _lowState.imu.quaternion[0] = msg.orientation.w;
    _lowState.imu.quaternion[1] = msg.orientation.x;
    _lowState.imu.quaternion[2] = msg.orientation.y;
    _lowState.imu.quaternion[3] = msg.orientation.z;

    _lowState.imu.gyroscope[0] = msg.angular_velocity.x;
    _lowState.imu.gyroscope[1] = msg.angular_velocity.y;
    _lowState.imu.gyroscope[2] = msg.angular_velocity.z;
    
    _lowState.imu.accelerometer[0] = msg.linear_acceleration.x;
    _lowState.imu.accelerometer[1] = msg.linear_acceleration.y;
    _lowState.imu.accelerometer[2] = msg.linear_acceleration.z;

    // lastRecv = clock;
}

void IOROS::FRhipCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[0].mode = msg.mode;
    _lowState.motorState[0].q = msg.q;
    _lowState.motorState[0].dq = msg.dq;
    _lowState.motorState[0].tauEst = msg.tauEst;
}

void IOROS::FRthighCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[1].mode = msg.mode;
    _lowState.motorState[1].q = msg.q;
    _lowState.motorState[1].dq = msg.dq;
    _lowState.motorState[1].tauEst = msg.tauEst;
}

void IOROS::FRcalfCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[2].mode = msg.mode;
    _lowState.motorState[2].q = msg.q;
    _lowState.motorState[2].dq = msg.dq;
    _lowState.motorState[2].tauEst = msg.tauEst;
}

void IOROS::FLhipCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[3].mode = msg.mode;
    _lowState.motorState[3].q = msg.q;
    _lowState.motorState[3].dq = msg.dq;
    _lowState.motorState[3].tauEst = msg.tauEst;
}

void IOROS::FLthighCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[4].mode = msg.mode;
    _lowState.motorState[4].q = msg.q;
    _lowState.motorState[4].dq = msg.dq;
    _lowState.motorState[4].tauEst = msg.tauEst;
}

void IOROS::FLcalfCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[5].mode = msg.mode;
    _lowState.motorState[5].q = msg.q;
    _lowState.motorState[5].dq = msg.dq;
    _lowState.motorState[5].tauEst = msg.tauEst;
}

void IOROS::RRhipCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[6].mode = msg.mode;
    _lowState.motorState[6].q = msg.q;
    _lowState.motorState[6].dq = msg.dq;
    _lowState.motorState[6].tauEst = msg.tauEst;
}

void IOROS::RRthighCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[7].mode = msg.mode;
    _lowState.motorState[7].q = msg.q;
    _lowState.motorState[7].dq = msg.dq;
    _lowState.motorState[7].tauEst = msg.tauEst;
}

void IOROS::RRcalfCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[8].mode = msg.mode;
    _lowState.motorState[8].q = msg.q;
    _lowState.motorState[8].dq = msg.dq;
    _lowState.motorState[8].tauEst = msg.tauEst;
}

void IOROS::RLhipCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[9].mode = msg.mode;
    _lowState.motorState[9].q = msg.q;
    _lowState.motorState[9].dq = msg.dq;
    _lowState.motorState[9].tauEst = msg.tauEst;
}

void IOROS::RLthighCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[10].mode = msg.mode;
    _lowState.motorState[10].q = msg.q;
    _lowState.motorState[10].dq = msg.dq;
    _lowState.motorState[10].tauEst = msg.tauEst;
}

void IOROS::RLcalfCallback(const unitree_legged_msgs::MotorState& msg)
{
    _lowState.motorState[11].mode = msg.mode;
    _lowState.motorState[11].q = msg.q;
    _lowState.motorState[11].dq = msg.dq;
    _lowState.motorState[11].tauEst = msg.tauEst;
}

#endif  // COMPILE_WITH_ROS