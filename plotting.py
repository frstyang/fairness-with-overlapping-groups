import matplotlib.pyplot as plt
import numpy as np
import pickle

def capitalize(string):
    return str.upper(string[0])+str[1:]

def slope(x1,x2,y1,y2):
    if x1 == x2 and y1==y2:
        return 0
    if x1 == x2 and y1<y2:
        return np.inf
    if x1 == x2 and y1>y2:
        return -np.inf
    sl=(y2-y1)/(x2-x1)
    return sl

def cvxhull(x,y):
    hull = []
    argsort = np.argsort(x)
    for i in range(len(x)):
        idx = argsort[i]
        if i > 0 and y[idx] > hull[-1][-1]:
            continue
        hull.append([x[idx], y[idx]])
    x, y = zip(*hull)
    new_x, new_y = [], []
    for i in range(len(x)):
        if i > 0:
            x_ta, y_ta = x[i], y[i]
            n,j=len(new_x), 0
            while j<n-1 and slope(new_x[-j-1], x_ta, new_y[-j-1], y_ta)\
                          < slope(new_x[-j-2], new_x[-j-1], new_y[-j-2], new_y[-j-1]):
                j+=1
            new_x = new_x[:n-j]
            new_y = new_y[:n-j]
        new_x.append(x[i])
        new_y.append(y[i])
                
    return np.array(new_x), np.array(new_y)

def plot_results(ax,results,dset,train=False, plottype=0, permute=True):
    # permute 
    if permute:
        results[0], results[2] = results[2], results[0]
    for result in results:
        metric_keys = result[0]['metrics'].keys()
        zipped_metrics = {}
        for point in result:
            name = point['method']
            for key in metric_keys:
                if key in zipped_metrics:
                    zipped_metrics[key].append(point['metrics'][key])
                else:
                    zipped_metrics[key] = [point['metrics'][key]]
        if not train:
            print("-------------------------------")
            print(f"Method: {name}")
            print(f"Average training time: {np.mean(zipped_metrics['train_time']):.5f}s")
            print(f"Average prediction time: {np.mean(zipped_metrics['predict_time']):.5f}s")
        prefix = 'train_' if train else ''
        if plottype==0:
            viols = zipped_metrics[prefix+'ind_viol']
        if plottype==1:
            viols = zipped_metrics[prefix+'ind_viol']
        if plottype==2:
            viols = zipped_metrics[prefix+'gerry_viol']
            
        accs = zipped_metrics[prefix+'accuracy']
        v,e = cvxhull(viols, 1-np.array(accs))
        ax.plot(v,e, linewidth=3, label=name)
    if plottype == 0:
        ax.set_xlabel('Independent group fairness violation', fontsize=12)
    if plottype == 1:
        ax.set_xlabel('Independent EO group fairness violation', fontsize=12)
    ax.set_ylabel('Classification error', fontsize=12)
    ax.legend()
    ax.set_title(f'{prefix}{str.upper(dset[0])}{dset[1:]}', fontsize=17)

if __name__ == '__main__':
    print('Enter name for saved figure:')
    name = input()
    print('Enter what exp to plot (0 for ind, 1, for eo, 2 for gerry)')
    exps = ['ind', 'eo']
    plottype = int(input())
    if plottype in [0,1]:
        fig, axes = plt.subplots(2,4,figsize=(25, 11))
        for i,dset in enumerate(['communities', 'adult', 'german', 'lawschool']):
            print('======================================')
            print(f'Plotting {dset} results....')
            with open(f'results/{dset}_{exps[plottype]}.pkl', 'rb') as f:
                results = pickle.load(f)
            plot_results(axes[0,i], results, dset, plottype=plottype)
            plot_results(axes[1,i], results, dset, True, plottype=plottype)
    if plottype == 2:
        fig, axes = plt.subplots(2,3, figsize=(18.5, 11))
        for i, dset in enumerate(['adult', 'german', 'lawschool']):
            print('======================================')
            print(f'Plotting {dset} results....')
            with open(f'results/{dset}_gerry.pkl', 'rb') as f:
                results = pickle.load(f)
            plot_results(axes[0,i], results, dset, gerry=True)
            plot_results(axes[1,i], results, dset, True, gerry=True)
    fig.savefig(f"plots/{name}")