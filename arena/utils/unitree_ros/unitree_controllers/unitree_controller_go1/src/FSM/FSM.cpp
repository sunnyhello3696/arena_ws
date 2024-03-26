/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include "FSM/FSM.h"
#include <iostream>

FSM::FSM(CtrlComponents *ctrlComp)
    :_ctrlComp(ctrlComp){

    if(ctrlComp->params){
        _targetState = (FSMStateName) ctrlComp->params->targetState;
    }

    _stateList.invalid = nullptr;
    _stateList.passive = new State_Passive(_ctrlComp);
    _stateList.fixedStand = new State_FixedStand(_ctrlComp);
    _stateList.freeStand = new State_FreeStand(_ctrlComp);
    _stateList.trotting = new State_Trotting(_ctrlComp);
    _stateList.balanceTest = new State_BalanceTest(_ctrlComp);
    _stateList.swingTest = new State_SwingTest(_ctrlComp);
    _stateList.stepTest = new State_StepTest(_ctrlComp);
    _stateList.danger = new State_Danger(_ctrlComp);
    

    #ifdef COMPILE_WITH_MOVE_BASE
        _stateList.moveBase = new State_move_base(_ctrlComp);
    #endif  // COMPILE_WITH_MOVE_BASE

    initialize();
    
}

FSM::~FSM(){
    _stateList.deletePtr();
}

void FSM::initialize(){

    _currentState = _stateList.passive;
    _currentState -> enter();
    _nextState = _currentState;
    _mode = FSMMode::NORMAL;
}

void FSM::run(){
    _startTime = getSystemTime();
    _ctrlComp->sendRecv();
    _ctrlComp->runWaveGen();
    _ctrlComp->estimator->run();
    if(!checkSafty()){
        // _ctrlComp->ioInter->setPassive(); nice safety
        _nextState = _stateList.danger;
        _mode = FSMMode::CHANGE;
    }

    if(_mode == FSMMode::NORMAL){
        _currentState->run();
        _nextStateName = _currentState->checkChangeOverride(_targetState);
        if(_nextStateName != _currentState->_stateName){
            _mode = FSMMode::CHANGE;
            _nextState = getNextState(_nextStateName);
        }
    }
    else if(_mode == FSMMode::CHANGE){
        if(_nextState->_stateName != _currentState->_stateName){
            std::cout << "Switched from " << _currentState->_stateNameString << " to " << _nextState->_stateNameString << std::endl;
            _currentState->exit();
        }
        _currentState = _nextState;
        _currentState->enter();
        _mode = FSMMode::NORMAL;
        _currentState->run();
    }

    absoluteWait(_startTime, (long long)(_ctrlComp->dt * 1000000));
}

FSMState* FSM::getNextState(FSMStateName stateName){
    switch (stateName)
    {
    case FSMStateName::INVALID:
        return _stateList.invalid;
        break;
    case FSMStateName::PASSIVE:
        return _stateList.passive;
        break;
    case FSMStateName::FIXEDSTAND:
        return _stateList.fixedStand;
        break;
    case FSMStateName::FREESTAND:
        return _stateList.freeStand;
        break;
    case FSMStateName::TROTTING:
        return _stateList.trotting;
        break;
    case FSMStateName::BALANCETEST:
        return _stateList.balanceTest;
        break;
    case FSMStateName::SWINGTEST:
        return _stateList.swingTest;
        break;
    case FSMStateName::STEPTEST:
        return _stateList.stepTest;
        break;
    case FSMStateName::DANGER:
        return _stateList.danger;
        break;
#ifdef COMPILE_WITH_MOVE_BASE
    case FSMStateName::MOVE_BASE:
        return _stateList.moveBase;
        break;
#endif  // COMPILE_WITH_MOVE_BASE
    default:
        return _stateList.invalid;
        break;
    }
}

bool FSM::checkSafty(){
    
    // The angle with z axis less than 60 degree
    if(_ctrlComp->lowState->getRotMat()(2,2) < 0.5 ){
        _safetyTimeout = getSystemTime() + 1e6; // 1s  
        // std::cout << "active danger " << _ctrlComp->lowState->getRotMat()(2,2) << std::endl;
        return false;
    }
    //keep waiting
    else if(getSystemTime() < _safetyTimeout){
        // std::cout << "latent danger, remaining " << (_safetyTimeout - getSystemTime()) << "us" << std::endl;
        return false;
    }
    else{
        return true;
    }
}