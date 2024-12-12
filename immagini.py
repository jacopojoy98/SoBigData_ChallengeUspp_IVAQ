import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import matplotlib
import os

matplotlib.rcParams['axes.linewidth'] = 1
SIZE_DEFAULT = 15
SIZE_LARGE = 20
plt.rc("font", family="Times New Roman")  # controls default font
plt.rc("font", weight="bold")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)


from scipy.stats.qmc import LatinHypercube, Sobol, Halton, scale

# Define parameters for the design space
def sobsamp():
    n_samples = 50
    dim = 2
    l_bounds, u_bounds = [0, 0], [10,10]

    sobol = Sobol(d=dim, scramble=True)  # Scramble adds some randomness to avoid regular patterns
    sobol_samples = scale(sobol.random_base2(m=6)[:n_samples], l_bounds, u_bounds)
    array = np.zeros((10,10))
    for [i,x_r] in enumerate(range(10)):
        x_samples = [a for a in sobol_samples if x_r<a[0]<x_r+1]
        for [j,y_r] in enumerate(range(10)):
            samples = [a for a in x_samples if y_r<a[1]<y_r+1]
            if samples:
                array[i][j]=1
    
    return array, sobol_samples


def random_flight_left_to_right(matrix):
    movement = 1
    x = 0
    y = 0
    series = [(0,0)]
    lenght = len(matrix)
    width = len(matrix[0]) 
    while y<width:
        y_mov = random.choice([1,1,1,-1,-1,0,0,0])
        y_ = y + y_mov
        if y_<0:
            continue
        y=y_
        x = x + movement
        if x == lenght or x == 0:
            movement = -movement
        
        if y<=9 and x<=9:
            series.append((x,y))
    # print(series)
    return series

def case_custom(i,j,path):
    if i==0 or i==9 or j==0 or j==9:
        return True
    else:
        return False

def case_1b(i,j,path):
    if (i+j)%2==0:
        return True
    else:
        return False

def case_path(i,j,path):
    if (j,i) in path:
        return True
    else:
        return False

def case_points(i,j,path):
    if path[j][i]!=0:
        return True
    else:
        return False

def case_zed(i,j,path):
    if path[j][i]!=0:
        return True
    else:
        return False

def case_1(i,j,path):
    if i>=2 and i<=7 and j>=2 and j<=7:
        return True
    else:
        return False
    
def case_0(i,j,path):
    return True

def image(title):
    # Define the main grid dimensions and sub-square count per cell
    main_grid_size = (10, 10)
    condition = globals()["case_"+title]
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    path = random_flight_left_to_right(np.zeros(main_grid_size))
    rarray, samples = sobsamp()
    if title in ["points","zed"]:
        path = rarray
    for i in range(main_grid_size[0]):
        for j in range(main_grid_size[1]):
            if condition(i,j,path):
                color = 'cornflowerblue'
            else:
                color = 'lightgray'
            rect = patches.Rectangle((j, main_grid_size[0] - i - 1), 1, 1, linewidth=0, edgecolor=None, facecolor=color)
            ax.add_patch(rect)

    # Define the position of the transimter
    x = random.randint(0,19)/2
    y = random.randint(0,19)/2
    # if title =="random":
    #     x=12
    #     y=3
    # if title == "0":
    #     x=11
    #     y=9
    # if title == "1":
    #     x=15
    #     y=5
    # if title == "1b":
    #     x=5
    #     y=1
    # if title == "custom":
    #     x = 11
    #     y = 1 
    # Draw the sub-square without edges to make them appear seamless within the main cell
    rect = patches.Rectangle((x, y), 0.5, 0.5, linewidth=0, edgecolor=None, facecolor='red')
    ax.add_patch(rect)
    if title in ["points","zed"]:
        ax.scatter(samples[:, 0],10 - samples[:, 1], color='yellow', marker='o', s=10, alpha=0.9)
    for i in range(main_grid_size[0]):
        for j in range(main_grid_size[1]):
            # Draw the main cell border with visible edges
            main_rect = patches.Rectangle((j, main_grid_size[0] - i - 1), 1, 1, linewidth=1, edgecolor='white', facecolor='none')
            ax.add_patch(main_rect)

    # Add a label overlay at the top
    # ax.text(5, 10.5,"case" + title, ha='center', va='center', fontsize=16, color='white', fontweight='bold',
    #         bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.5'))

    # Set the limits, aspect, and hide axes
    p=0.1
    ax.set_xlim(0-p, main_grid_size[1]-p)
    ax.set_ylim(0-p, main_grid_size[0]-p)
    ax.set_aspect('equal')
    ax.axis('off')
    # fig.tight_layout()
    # Display the plot
    plt.savefig("C:\\Users\\jacop\\Desktop\\SoBigData\\IMG\\"+title+".png", bbox_inches='tight')
    plt.close()

def statistics(experiment,case):

    # torch.save(model.state_dict(),"modelcase_"+case+"nb"+str(num_beacons)+".pt")
    tlos = np.loadtxt(experiment+"\\testlossescase_"+case+"nb"+str(num_beacons)+".txt")
    trlos = np.loadtxt(experiment+"\\trainlossescase_"+case+"nb"+str(num_beacons)+".txt")
    Maxlos = np.loadtxt(experiment+"\\Maxtestlossescase_"+case+"nb"+str(num_beacons)+".txt")
    finallos = np.loadtxt(experiment+"\\final_eval_distances_"+case+"nb"+str(num_beacons)+".txt")

    IMGdir = os.path.join(experiment,"IMG")
    os.makedirs(IMGdir, exist_ok=True)
    x_axis = np.arange(0,len(tlos),1)
    yaxis = tlos
    inte = 2
    # plt.figure(figsize=(16,10))
    plt.plot(x_axis,yaxis)
    plt.title(case+"\n Test loss")
    # plt.show()
    plt.xlabel("epochs")
    plt.ylabel("Distance")
    plt.tight_layout()
    savepath=os.path.join(IMGdir,"testloss"+case+"nb"+str(num_beacons)+".png")
    plt.savefig(savepath)
    plt.close()

    # plt.figure(figsize=(16,10))
    plt.plot(x_axis,trlos)
    plt.title(case+"\n Train loss")
    # plt.show()
    plt.xlabel("epochs")
    plt.ylabel("Distance")
    plt.tight_layout()
    savepath=os.path.join(IMGdir,"trainloss"+case+"nb"+str(num_beacons)+".png")
    plt.savefig(savepath)
    plt.close()


    # plt.figure(figsize=(16,10))
    plt.plot(x_axis,Maxlos)
    plt.title(case+"\n max Distance" )
    plt.xlabel("epochs")
    plt.ylabel("Distance")
    plt.tight_layout()
    # plt.show()
    savepath=os.path.join(IMGdir,"maxloss"+case+"nb"+str(num_beacons)+".png")
    plt.savefig(savepath)
    plt.close()

    # plt.figure(figsize=(16,10))
    plt.hist(finallos)
    mean = np.mean(finallos)
    ylim = plt.ylim()
    plt.axvline(mean,ymin=0,ymax=ylim[1],color = 'red')
    plt.xlabel("Distance")
    plt.ylabel("")
    plt.text(mean+0.01,int(0.9*ylim[1]), f"mean = {mean:.4f} m")
    plt.title(case+"\n Distance distribution")
    plt.tight_layout()
    savepath=os.path.join(IMGdir,"Distrloss"+case+"nb"+str(num_beacons)+".png")
    plt.savefig(savepath)
    plt.close()




# for condition in ['case_custom', 'case_0','case_1','case_1b']:
#     image(condition=globals()[condition],title=condition)


num_beacons = 1
base = "C:\\Users\\jacop\\Desktop\\SoBigData\\eltr"
experiments = [os.path.join(base,a) for a in os.listdir(base) if os.path.isdir(os.path.join(base,a))]

for case in ["0","1","1b","custom","path","points","zed"]:
    image(case)
exit()
case = "random"
for experiment in experiments:

    statistics(experiment,case)


