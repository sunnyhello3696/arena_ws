/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#include "FSM/State_Danger.h"

State_Danger::State_Danger(CtrlComponents *ctrlComp)
                :FSMState(ctrlComp, FSMStateName::DANGER, "danger"){}

void State_Danger::enter(){
    for(int i=0; i<4; i++){
        if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
            _lowCmd->setSimStanceGain(i);
        }
        else if(_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT){
            _lowCmd->setRealStanceGain(i);
        }
        _lowCmd->setZeroDq(i);
        _lowCmd->setZeroTau(i);
    }
    for(int i=0; i<12; i++){
        _lowCmd->motorCmd[i].mode = 0x0A;
        _lowCmd->motorCmd[i].q = _targetPos[i % 3];
    }
    _ctrlComp->setAllStance();
}

FSMStateName State_Danger::checkChange(){
    return FSMStateName::PASSIVE;
}
