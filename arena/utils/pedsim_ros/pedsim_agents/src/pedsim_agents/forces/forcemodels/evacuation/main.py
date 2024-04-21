from pedsim_agents.forces import Forcemodel, Forcemodels

import pedsim_msgs.msg

@Forcemodels.register(Forcemodel.Name.EVACUATION)
class Plugin_Evacuation(Forcemodel):
    def __init__(self):
        global Diff_Equ, Room, np, leap_frog

        from .Integrators import leap_frog
        from .diff_equation import Diff_Equ
        from .Room import Room
        import numpy as np

    def compute(self, in_data, work_data):
        '''
        "num_steps" is the duration the simulation will run (recommended:1000)

       "method" is the method of integration. You should use leap_frog even though it will often explode
        since more relaible methods of integration like ode45 and monto carlo take a lot a computational power.
        '''
        N = len(in_data.agents)                            # quantity of pedestrians aka the number of agents that are currently in the simulation
        
        tau = 1                                         # time-step (s), doesn't seem to affect calculation in our case
        num_steps = 2                                   # the number of force-calculation steps the simulation should go through, "2" equals one step
        room_size = 40                                 # size of square room (m), TODO: integrate real value
        method = leap_frog                              # method used for integration -> leap-frog was the GoTo solution in the original project
        radii = 0.3 * np.ones(N)                        # radii of pedestrians (m) -> was "0.4 * (np.ones(self.N)*variation).squeeze()" before
        m = 180 * np.ones(N)                             # mass of pedestrians (kg) -> was "80 * (np.ones(self.N)*variation).squeeze()" before

        v = np.zeros((2, N, num_steps))                 # Three dimensional array of velocity, not used in leap frog strangely
        y = np.zeros((2, N, num_steps))                 # Three dimensional array of place: x = coordinates(2 dims), y = Agent (N dims), z=Time (2 dims)
        for i in range(N):
            pos_x = in_data.agents[i].pose.position.x
            pos_y = in_data.agents[i].pose.position.y
            pos = [pos_x, pos_y] 
            y[:, i, 0] = pos                            # z = 0 -> start position

        """
        obstacles = np.zeros((2, 2, len(data.line_obstacles)))
        for i in range(len(data.line_obstacles)):
            start_pos = [data.line_obstacles[i].start.x, data.line_obstacles[i].start.y]
            end_pos = [data.line_obstacles[i].end.x, data.line_obstacles[i].end.y]
            obstacles[:, :, i] = [start_pos, end_pos]
        """

        room = Room("arena", room_size, in_data.walls)                 # kind of room the simulation runs in, added arena room

        """if(data.line_obstacles is not None):
            for i, obstacle in enumerate(data.line_obstacles):
                print(i, obstacle)"""
        
        #create differential equation object
        diff_equ = Diff_Equ(N, room_size, tau, room, radii, m)  # initialize Differential equation

        # calls the method of integration with the starting positions, diffequatial equation, number of steps, and delta t = tau
        y, agents_escaped, forces = method(y[:, :, 0], v[:, :, 0], diff_equ.f, num_steps, tau, room)
        # forces is of shape (2, N, 2)

        work_data.force[:,[0,1]] = forces[[0,1],:,0].T

