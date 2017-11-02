import numpy as np
from matplotlib import pyplot as plt

# Complete the following to make the plot
if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    # Get a colour map
    cmap = plt.get_cmap('YlOrRd')

    # Define our colour indexes u-g and r-i
    ug = data['u']-data['g']
    ri = data['r']-data['i']
    # Make a redshift array
    rs = data['redshift']
    print(min(ug), max(ug), min(ri), max(ri))
    # Create the plot with plt.scatter and plt.colorbar
    plt.scatter(x = ug, y = ri, c = rs, s=10, cmap='YlOrRd', lw=0)  
    plt.colorbar()
    
    # Define your axis labels and plot title

    plt.xlabel("Colour index u-g")
    plt.ylabel("Colour index r-i")
    plt.title("Redshift (colour) u-g versus r-i")
    # Set any axis limits
    
    plt.xlim(-0.5,2.5)
    plt.ylim(-0.5,1)

    plt.show()