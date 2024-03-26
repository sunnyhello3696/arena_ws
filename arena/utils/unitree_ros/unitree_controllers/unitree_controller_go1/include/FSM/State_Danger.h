/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#ifndef STATE_DANGER_H
#define STATE_DANGER_H

#include "FSM/FSMState.h"

class State_Danger : public FSMState{
public:
    State_Danger(CtrlComponents *ctrlComp);
    ~State_Danger(){}
    void enter();
    void run(){}
    void exit(){}
    FSMStateName checkChange();
private:
    float _targetPos[3] = {
        +0*M_PI/180, +90*M_PI/180, -132*M_PI/180
        //bend hip [-40°,+40°], bend leg [-45°,+218°], bend foot [-132°,-24°]
    };
};

#endif  // STATE_CURLEDUP_H