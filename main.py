from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile
from trainingPPO import Simulation
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='config.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Visualization = Visualization(
        plot_path, 
        dpi=96 
    )
         
    Simulation = Simulation(
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_actions'],
        3
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'],train= 0)  # run the simulation
    print('Simulation time:', simulation_time, 's')
 
    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
    
    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')
