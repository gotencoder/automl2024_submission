import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import ticker
import numpy as np
import h5py
from matplotlib import gridspec




def plot_3d(points, points_color, title, show_plot=False, save_fig=False, save_path=None, s=1):
    x, y, z = points[:,0], points[:,1], points[:,2]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=12)
    col = ax.scatter(x, y, z, c=points_color, s=s, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.tight_layout()
    if save_fig==True:
        if save_path==None:
            raise Exception('Please set a save path')
        plt.savefig(save_path,format='pdf',bbox_inches=None)
    if show_plot==True:
        plt.show()


def add_2d_scatter(ax, points, points_color, title=None, s=50):
    x, y = points[:,0], points[:,1]
    ax.scatter(x, y, c=points_color, s=s, alpha=0.8)
    ax.set_title(title)
    


def plot_2d(points, points_color, title, show_plot=False, save_fig=False, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.tight_layout()
    if save_fig==True:
        if save_path==None:
            raise Exception('Please set a save path')
        plt.savefig(save_path,format='pdf',bbox_inches=None)
    if show_plot==True:
        plt.show()




#Return the x,y,z list in an order from smallest z to largest z - or x or y
def colour_object(x,y,z,constrain_z=False,zlim=None,include_region=False):
    if constrain_z==True:
        exclude_index = np.where((z >-zlim) & (z <zlim))
        if include_region==False:
            z = np.delete(z,exclude_index)
            x = np.delete(x,exclude_index)
            y = np.delete(y,exclude_index)
        else:
            z = z[exclude_index]
            y = y[exclude_index]
            x = x[exclude_index]    
    
    sorted_z_index = np.argsort(z)
    x_out, y_out, z_out = x[sorted_z_index], y[sorted_z_index], z[sorted_z_index]

    return x_out, y_out, z_out


#This function is used to index points via a rotational axis - either theta or phi directions
def colour_object_rotation(x,y,z,rotation_axis=None):
    r_val = np.sqrt(x**2 + y**2 + z**2)
    theta_val = np.arctan2(y,x)
    phi_val = np.arccos(z/r_val)

    #Now colour based on the theta values
    if rotation_axis=='theta':
        sorted_theta_index = np.argsort(theta_val)
    elif rotation_axis=='phi':
        sorted_theta_index = np.argsort(phi_val)
    else:
        raise ValueError('Please specify a rotational axis')   

    x_out, y_out, z_out = x[sorted_theta_index], y[sorted_theta_index], z[sorted_theta_index]

    return x_out, y_out, z_out

    

def save_2d_embedding(data_matrix,scores,filename=None):
    f = h5py.File(filename, 'w')
    
    # Create datasets for x and y
    f.create_dataset('x', data=data_matrix[:,0])
    f.create_dataset('y', data=data_matrix[:,1])
    f.create_dataset('scores', data=scores)
    f.close()



def single_plot_3d(latent_space, times, title=None, show_image=True, save_image=False, save_file=None, rotate_angles=None):
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 1, left=0.08, right=0.92, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

    ax1 = plt.subplot(gs[0], projection='3d')
    ax1.set_title(title)
    
    # Scatter plot the 3D embeddings
    scatter = ax1.scatter(latent_space[:, 0], latent_space[:, 1], latent_space[:, 2], c=times, marker='o', alpha=0.5)
    
    ax1.set_xlabel('z1')
    ax1.set_ylabel('z2')
    ax1.set_zlabel('z3')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    
    # Rotate the plot if rotation angles are provided
    if rotate_angles is not None:
        ax1.view_init(elev=rotate_angles[0], azim=rotate_angles[1])

    # Create a separate axis for the color bar
    cax = fig.add_axes([0.76, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    
    # Add the color bar to the separate axis
    colorbar = plt.colorbar(scatter, cax=cax)
    colorbar.set_label('Time')
    
    if save_image:
        plt.savefig(save_file, format='pdf')
        plt.close()

    if show_image:     
        plt.show()

