import matplotlib.pyplot as plt
import seaborn as sns

def generate_baseline_performance(labels, output_folder, filename):

    # change dataframe labels - no as 0 and yes as 1
    # labels = labels.replace('no', 0)
    # labels = labels.replace('yes', 1)

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
    plt.title(filename+'\'s Recall vs Number of samples')
    plt.savefig(output_folder+'/baseline_performance_'+filename+'.png')

def create_active_learning_graph(recalls, baseline_recall, filename, output_folder):
    plt.clf()
    baseline_recall = baseline_recall[:len(recalls)]
    plt.plot(recalls, label='Active Learner')
    plt.plot(baseline_recall, label=filename+'\'s recall')
    plt.xlabel('Number of samples')
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Learner Recall vs Baseline Recall')
    plt.savefig(output_folder+'/active_learning_performance_'+filename+'.png')

def create_stat_graph(per_50, per_25, per_75, baselines, filename, output_folder, iterations, no_statistical_validation):
    plt.clf()
    # plt.errorbar(range(len(mean)), mean, yerr=std, label='Active Learner')
    sns.set_style("whitegrid")
    sns.lineplot(x=range(len(per_50)), y=per_50, label='50th percentile', linestyle='solid')
    sns.lineplot(x=range(len(per_25)), y=per_25, label='25th percentile', linestyle='dotted')
    sns.lineplot(x=range(len(per_75)), y=per_75, label='75th percentile', linestyle='dashed')
    plt.fill_between(range(len(per_25)), per_25, per_75, alpha=0.2)
    sns.lineplot(x=range(len(baselines)), y=baselines, label='Baseline', linestyle='dashdot', palette='Reds')
    plt.xlabel('Per iteration recall variance ('+str(iterations)+' iterations over '+str(no_statistical_validation)+' runs)')
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Learner Recall variance')
    plt.savefig(output_folder+'/statistical_performance_'+filename+'.png')
    return

def create_combined_graph(fifties, twenty_fives, seventy_fives, baseline_recalls, starts, filename, output_folder, iterations, no_statistical_validation):
    # plt.clf()
    # for i, start in enumerate(starts):
    #     plt.errorbar(range(len(means[i])), means[i], yerr=stds[i], label='Learner start at - '+str(start))
    # plt.xlabel('Per iteration recall variance ('+str(iterations)+' iterations over '+str(no_statistical_validation)+' runs)')
    # plt.ylabel('Recall')
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.savefig(output_folder+'/combined_statistical_performance_'+filename+'.png')

    # convert this into a seabron line graph plot
    plt.clf()
    sns.set_style("whitegrid")

    pallettes = ['Oranges', 'Blues', 'Greens']

    for i, start in enumerate(starts):
        sns.lineplot(x=range(len(fifties[i])), y=fifties[i], label='50th percentile - Learner start at - '+str(start), linestyle='solid')
        sns.lineplot(x=range(len(twenty_fives[i])), y=twenty_fives[i], label='25th percentile - Learner start at - '+str(start), linestyle='dotted')
        sns.lineplot(x=range(len(seventy_fives[i])), y=seventy_fives[i], label='75th percentile - Learner start at - '+str(start), linestyle='dashed')
        plt.fill_between(range(len(twenty_fives[i])), twenty_fives[i], seventy_fives[i], alpha=0.2)
        sns.color_palette(palette=pallettes[i])
    
    sns.lineplot(x=range(len(baseline_recalls)), y=baseline_recalls, label='Baseline', linestyle='dashdot')

    plt.xlabel('Per iteration recall variance ('+str(iterations)+' iterations over '+str(no_statistical_validation)+' runs)')
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Learner Recall starting with multiple initial yes samples')
    plt.savefig(output_folder+'/combined_statistical_performance_'+filename+'.png')
    return