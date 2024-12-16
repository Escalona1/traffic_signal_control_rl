import matplotlib.pyplot as plt
import numpy as np
import os

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi
    

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
         
        """
        Produce a plot of performance of the agent over the session averaged over x episodes
        and save the relative data to txt.
        """

        x = 70
        # Calculate the moving average
        averaged_data = [np.mean(data[i:i+x]) for i in range(0, len(data), x)]
        
        # Determine min and max values for the plot
        min_val = min(averaged_data)
        max_val = max(averaged_data)

        plt.rcParams.update({'font.size': 24})  # Set bigger font size

        # Plot the averaged data
        plt.plot(averaged_data, marker='o', linestyle='-', color='b', label=f'Mean of {x} episodes')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        plt.legend()
        
        # Adjust the figure size and save the plot
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        # Save the averaged data to a file
        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                file.write("%s\n" % value)