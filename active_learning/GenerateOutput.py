import matplotlib.pyplot as plt

def generate_baseline_performance(labels, output_folder, filename):

    # change dataframe labels - no as 0 and yes as 1
    labels = labels.replace('no', 0)
    labels = labels.replace('yes', 1)

    #calculate recall using n samples where n ranges from 1 to len(labels)

    recall = []

    for i in range(1, len(labels)):
        recall.append(labels[:i].sum()/labels.sum())
    
    create_baseline_graph(recall, filename, output_folder)

    return recall

def create_baseline_graph(recall, filename, output_folder):
    plt.clf()
    plt.plot(recall)
    plt.xlabel('Number of samples')
    plt.ylabel('Recall')
    plt.title('Recall vs Number of samples '+filename)
    plt.savefig(output_folder+'baseline_performance_'+filename+'.png')

def create_active_learning_graph(recalls, baseline_recall, filename, output_folder):
    plt.clf()
    baseline_recall = baseline_recall[:len(recalls)]
    plt.plot(recalls, label='Active Learner')
    plt.plot(baseline_recall, label='Baseline')
    plt.xlabel('Number of samples')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig(output_folder+'active_learning_performance_'+filename+'.png')

def create_stat_graph(mean, std, filename, output_folder, iterations, no_statistical_validation):
    plt.clf()
    plt.errorbar(range(len(mean)), mean, yerr=std, label='Active Learner')
    plt.xlabel('Per iteration recall variance ('+str(iterations)+' iterations over '+str(no_statistical_validation)+' runs)')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig(output_folder+'statistical_performance_'+filename+'.png')
    return