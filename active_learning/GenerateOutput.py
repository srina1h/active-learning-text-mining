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
    plt.savefig(output_folder+'/baseline_performance_'+filename+'.pdf')

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
    plt.savefig(output_folder+'/active_learning_performance_'+filename+'.pdf')

def create_stat_graph(per_50, per_25, per_75, baselines, filename, output_folder, iterations, no_statistical_validation, labeling_budget):
    plt.clf()
    sns.set_style("whitegrid")

    x_coor = range(labeling_budget, len(per_25) + labeling_budget)
    plt.plot(x_coor, per_50, label='50th percentile', linestyle='solid', color='royalblue')
    plt.plot(x_coor, per_25, label='25th percentile', linestyle='dotted', color='skyblue')
    plt.plot(x_coor, per_75, label='75th percentile', linestyle='dashed', color='skyblue')

    plt.fill_between(x_coor, per_25, per_75, alpha=0.2, color='skyblue')
    plt.plot(baselines, label='Baseline', linestyle='dashdot', color='red')
    plt.xlabel('Per iteration recall variance ('+str(iterations)+' iterations over '+str(no_statistical_validation)+' runs)')
    plt.axvline(x=labeling_budget, color='black', linestyle='dotted', alpha=0.5)
    plt.text(labeling_budget, 0.1, 'Labeling Budget', rotation=90)
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.xscale('log')
    plt.legend()
    plt.title('Learner Recall variance')
    plt.savefig(output_folder+'/statistical_performance_'+filename+'.pdf', bbox_inches='tight')
    return

def create_combined_graph(fifties, twenty_fives, seventy_fives, baseline_recalls, starts, filename, output_folder, iterations, no_statistical_validation, top_tfidf):
    plt.clf()
    sns.set_style("whitegrid")

    colors = ['royalblue', 'orange', 'green']
    colorbands = ['skyblue', 'gold', 'mediumaquamarine']

    for i, start in enumerate(starts):
        x_coor = range(start, len(twenty_fives[i]) + start)
        plt.plot(x_coor, fifties[i], label='50th percentile - Learner start at - '+str(start), linestyle='solid', color=colors[i])
        plt.plot(x_coor, twenty_fives[i], label='25th percentile - Learner start at - '+str(start), linestyle='dotted', color=colorbands[i])
        plt.plot(x_coor, seventy_fives[i], label='75th percentile - Learner start at - '+str(start), linestyle='dashed', color=colorbands[i])

        plt.fill_between(x_coor, twenty_fives[i], seventy_fives[i], alpha=0.2, color=colorbands[i])

        plt.axvline(x=start, color='black', linestyle='dotted', alpha=0.5)
        plt.text(start, 0.1, 'Labeling Budget - '+str(start), rotation=90)

    plt.plot(baseline_recalls, label='Baseline', linestyle='dashdot', color='red')

    plt.xlabel('Per iteration recall variance ('+str(iterations)+' iterations over '+str(no_statistical_validation)+' runs)')
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.xscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Learner Recalls with '+str(top_tfidf)+' TF-IDF features')
    plt.savefig(output_folder+'/combined_statistical_performance_'+filename+'.pdf', bbox_inches='tight')
    return

def compare_models_graph(fifties, twenty_fives, seventy_fives, baseline_recalls, start, filename, output_folder, iterations, no_statistical_validation, top_tfidf, model_types):
    plt.clf()
    sns.set_style("whitegrid")

    colors = ['royalblue', 'green', 'orange']
    colorbands = ['skyblue', 'mediumaquamarine', 'gold']

    for i, model in enumerate(model_types):
        x_coor = range(start, len(twenty_fives[i]) + start)
        plt.plot(x_coor, fifties[i], label='50th percentile - '+str(model), linestyle='solid', color=colors[i])
        plt.plot(x_coor, twenty_fives[i], label='25th percentile - '+str(model), linestyle='dotted', color=colorbands[i])
        plt.plot(x_coor, seventy_fives[i], label='75th percentile - '+str(model), linestyle='dashed', color=colorbands[i])

        plt.fill_between(x_coor, twenty_fives[i], seventy_fives[i], alpha=0.2, color=colorbands[i])

        plt.axvline(x=start, color='black', linestyle='dotted', alpha=0.5)
        plt.text(start, 0.1, 'Labeling Budget - '+str(start), rotation=90)

    plt.plot(baseline_recalls, label='Baseline', linestyle='dashdot', color='red')

    plt.xlabel('Per iteration recall variance ('+str(iterations)+' iterations over '+str(no_statistical_validation)+' runs)')
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.xscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Comparing learners\' Recalls with '+str(top_tfidf)+' TF-IDF features')
    plt.savefig(output_folder+'/compare_model_performance_start_'+str(start)+filename+'.pdf', bbox_inches='tight')
    return