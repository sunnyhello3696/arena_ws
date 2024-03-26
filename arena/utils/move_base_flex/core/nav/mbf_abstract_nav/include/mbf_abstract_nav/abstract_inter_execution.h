/*
 *  Copyright 2018, Magazino GmbH, Sebastian Pütz, Jorge Santos Simón
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  abstract_inter_execution.h
 *
 *  authors:
 *    Sebastian Pütz <spuetz@uni-osnabrueck.de>
 *    Jorge Santos Simón <santos@magazino.eu>
 *
 */

#ifndef MBF_ABSTRACT_NAV__ABSTRACT_INTER_EXECUTION_H_
#define MBF_ABSTRACT_NAV__ABSTRACT_INTER_EXECUTION_H_

#include <map>
#include <string>
#include <vector>

#include <geometry_msgs/PoseStamped.h>

#include <mbf_abstract_core/abstract_inter.h>

#include <mbf_utility/robot_information.h>
#include <mbf_utility/navigation_utility.h>

#include "mbf_abstract_nav/abstract_execution_base.h"

namespace mbf_abstract_nav
{

/**
 * @defgroup inter_execution Inter Execution Classes
 * @brief The inter execution classes are derived from the AbstractInterExecution and extend the functionality.
 *        The base inter execution code is located in the AbstractInterExecution.
 */

/**
 * @brief The AbstractInterExecution class loads and binds the inter plugin. It contains a thread running
 *        the plugin in a cycle to plan and re-plan. An internal state is saved and will be pulled by the server, which
 *        controls the inter execution. Due to a state change it wakes up all threads connected to the
 *        condition variable.
 *
 * @ingroup abstract_server inter_execution
 */
  class AbstractInterExecution : public AbstractExecutionBase
  {
  public:

    //! shared pointer type to the @ref inter_execution "inter execution".
    typedef boost::shared_ptr<AbstractInterExecution > Ptr;

    /**
     * @brief Constructor
     * @param name Name of this execution
     * @param inter_ptr Pointer to the inter
     * @param robot_info Current robot state
     * @param config Initial configuration for this execution
     */
    AbstractInterExecution(const std::string& name,
                             const mbf_abstract_core::AbstractInter::Ptr& inter_ptr,
                             const mbf_utility::RobotInformation &robot_info,
                             const ros::Publisher &goal_pub,
                             const MoveBaseFlexConfig& config);

    /**
     * @brief Destructor
     */
    virtual ~AbstractInterExecution();

    /**
     * @brief Returns a new plan, if one is available.
     */
    std::vector<geometry_msgs::PoseStamped> getPlan() const;

    /**
     * @brief Returns the last time a valid plan was available.
     * @return time, the last valid plan was available.
     */
    ros::Time getLastValidPlanTime() const;

    /**
     * @brief Checks whether the patience was exceeded.
     * @return true, if the patience duration was exceeded.
     */
    bool isPatienceExceeded() const;

    /**
     * @brief Internal states
     */
    enum PlanningState
    {
      INITIALIZED,        ///< Inter initialized.
      STARTED,            ///< Inter started.
      PLANNING,           ///< Executing the plugin.
      FOUND_PLAN,         ///< Found a valid plan.
      MAX_RETRIES,        ///< Exceeded the maximum number of retries without a valid command.
      PAT_EXCEEDED,       ///< Exceeded the patience time without a valid command.
      NO_PLAN_FOUND,      ///< No plan has been found (MAX_RETRIES and PAT_EXCEEDED are 0).
      CANCELED,           ///< The inter has been canceled.
      STOPPED,            ///< The inter has been stopped.
      INTERNAL_ERROR,     ///< An internal error occurred.
      NO_GLOBAL_PLAN,     ///< Missing global plan input.
      EMPTY_GLOBAL_PLAN,  ///< Global plan is empty.
      INVALID_GLOBAL_PLAN ///< Global plan is invalid.
    };

    /**
     * @brief Returns the current internal state
     * @return the current internal state
     */
    PlanningState getState() const;

    /**
     * @brief Gets planning frequency
     */
    double getFrequency() const { return frequency_; };

    /**
     * @brief Gets computed costs
     * @return The costs of the computed path
     */
    double getCost() const;

    /**
     * @brief Cancel the inter execution. This calls the cancel method of the inter plugin.
     * This could be useful if the computation takes too much time, or if we are aborting the navigation.
     * @return true, if the inter plugin tries / tried to cancel the planning step.
     */
    virtual bool cancel();

    /**
     * @brief Sets a new goal pose for the inter execution
     * @param goal the new goal pose
     */
    void setNewGoal(const geometry_msgs::PoseStamped &goal);

    /**
     * @brief Sets a new start pose for the inter execution
     * @param start new start pose
     */
    void setNewStart(const geometry_msgs::PoseStamped &start);

    /**
     * @brief Sets a new star and goal pose for the inter execution
     * @param start new start pose
     * @param goal new goal pose
     */
    void setNewStartAndGoal(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal);

    /**
     * @brief Starts the inter execution thread with the given parameters.
     * @param start start pose for the planning
     * @param goal goal pose for the planning
     * @return true, if the inter thread has been started, false if the thread is already running.
     */
    bool start(const geometry_msgs::PoseStamped &start, const geometry_msgs::PoseStamped &goal);

    /**
     * @brief Is called by the server thread to reconfigure the controller execution, if a user uses dynamic reconfigure
     *        to reconfigure the current state.
     * @param config MoveBaseFlexConfig object
     */
    void reconfigure(const MoveBaseFlexConfig &config);

    /**
     * @brief Sets a new plan to the controller execution
     * @param plan A vector of stamped poses.
     * @param tolerance_from_action flag that will be set to true when the new plan (action) has tolerance
     * @param action_dist_tolerance distance to goal tolerance specific for this new plan (action)
     * @param action_angle_tolerance angle to goal tolerance specific for this new plan (action)
     */
    void setNewPlan(
      const std::vector<geometry_msgs::PoseStamped> &plan);

  protected:

    //! the local planer to calculate the robot trajectory
    mbf_abstract_core::AbstractInter::Ptr inter_;

    //! the name of the loaded inter plugin
    std::string plugin_name_;

    /**
     * @brief The main run method, a thread will execute this method. It contains the main inter execution loop.
     */
    virtual void run();

  private:

    /**
     * @brief calls the inter plugin to make a plan from the start pose to the goal pose.
     * @param start The start pose for planning
     * @param goal The goal pose for planning
     * @param plan The computed plan by the plugin
     * @param cost The computed costs for the corresponding plan
     * @param message An optional message which should correspond with the returned outcome
     * @return An outcome number, see also the action definition in the GetInterPath.action file
     */
    virtual uint32_t makePlan(
        const geometry_msgs::PoseStamped &start,
        const geometry_msgs::PoseStamped &goal,
        std::vector<geometry_msgs::PoseStamped> &plan,
        double &cost,
        std::string &message);

    /**
     * @brief Sets the internal state, thread communication safe
     * @param state the current state
     * @param signalling set true to trigger the condition-variable for state-update
     */
    void setState(PlanningState state, bool signalling = true);

    //! mutex to handle safe thread communication for the current state
    mutable boost::mutex state_mtx_;

    //! mutex to handle safe thread communication for the plan and plan-costs
    mutable boost::mutex plan_mtx_;

    //! mutex to handle safe thread communication for the global plan.
    mutable boost::mutex global_plan_mtx_;

    //! mutex to handle safe thread communication for the goal and start poses.
    mutable boost::mutex goal_start_mtx_;

    //! mutex to handle safe thread communication for the interpolating_ flag.
    mutable boost::mutex interpolating_mtx_;

    //! dynamic reconfigure mutex for a thread safe communication
    mutable boost::mutex configuration_mutex_;

    //! true, if a new goal pose has been set, until it is used.
    bool has_new_goal_;

    //! true, if a new start pose has been set, until it is used.
    bool has_new_start_;

    //! true, if a new plan is available. See hasNewPlan()!
    bool new_plan_;

    //! the last call start time, updated each cycle.
    ros::Time last_call_start_time_;

    //! the last time a valid plan has been computed.
    ros::Time last_valid_plan_time_;

    //! current interpolation result
    std::vector<geometry_msgs::PoseStamped> inter_plan_;

    //! current global plan cost
    double cost_;

    //! the current start pose used for planning
    geometry_msgs::PoseStamped start_;

    //! the current goal pose used for planning
    geometry_msgs::PoseStamped goal_;

    //! planning cycle frequency (used only when running full navigation; we store here for grouping parameters nicely)
    double frequency_;

    //! planning patience duration time
    ros::Duration patience_;

    //! planning max retries
    int max_retries_;

    //! main cycle variable of the execution loop
    bool interpolating_;

    //! robot frame used for computing the current robot pose
    std::string robot_frame_;

    //! the global frame in which the inter needs to plan
    std::string global_frame_;

    //! current internal state
    PlanningState state_;

    /**
     * @brief Returns true if a new plan is available, false otherwise! A new plan is set by another thread!
     * @return true, if a new plan has been set, false otherwise.
     */
    bool hasNewPlan();

    /**
     * @brief Gets the new available plan. This method is thread safe.
     * @return The plan
     */
    std::vector<geometry_msgs::PoseStamped> getNewPlan();

    //! the last set plan which is currently processed by the inter
    std::vector<geometry_msgs::PoseStamped> global_plan_;

    //! publisher for the global goal
    ros::Publisher global_goal_pub_;

  };

} /* namespace mbf_abstract_nav */

#endif /* MBF_ABSTRACT_NAV__ABSTRACT_INTER_EXECUTION_H_ */
