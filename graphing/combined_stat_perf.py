import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import seaborn as sns


def plot_combined_stat_perf(save_path=None):
    """
    The pickles are of the format list(list(numpy.ndarray))
    The first dimenion is number of models
    Second dimensions is the number of different labeling budgets
    Third dimension is the statistic in each iteration of learning
    """
    dataset_name = "Hall"
    tfidf_features = "50"
    starts = [8, 16, 32]
    pickle_path_50 = (
        "pickles/test_op_"
        + tfidf_features
        + "/"
        + dataset_name
        + "/recall_fifty_percentiles.pkl"
    )
    pickle_path_25 = (
        "pickles/test_op_"
        + tfidf_features
        + "/"
        + dataset_name
        + "/recall_twenty_five_percentiles.pkl"
    )
    pickle_path_75 = (
        "pickles/test_op_"
        + tfidf_features
        + "/"
        + dataset_name
        + "/recall_seventy_five_percentiles.pkl"
    )
    baseline_recalls = (
        "pickles/test_op_"
        + tfidf_features
        + "/"
        + dataset_name
        + "/baseline_recall.pkl"
    )

    # load the pickled data and print their types
    with open(pickle_path_50, "rb") as f:
        recall_50 = pickle.load(f)
    with open(pickle_path_25, "rb") as f:
        recall_25 = pickle.load(f)
    with open(pickle_path_75, "rb") as f:
        recall_75 = pickle.load(f)
    with open(baseline_recalls, "rb") as f:
        baseline_recalls = pickle.load(f)

    # print the types of the loaded data
    print(type(recall_50))
    print(len(recall_50))
    print(type(recall_50[1]))
    print(len(recall_50[1]))
    print(type(recall_50[1][1]))
    print(recall_50[1][1].shape)
    
    # print(recall_25[0][0][:10])
    # print(recall_50[0][0][:10])
    # print(recall_75[0][0][:10])

    # compare the three numpy arrays and check if they are the same

    # print(np.array_equal(recall_25[0][0], recall_50[0][0]))
    # print(np.array_equal(recall_25[0][0], recall_75[0][0]))
    # print(np.array_equal(recall_50[0][0], recall_75[0][0]))

    # models = ["GPM", "NB"]
    # for i in range(len(recall_50)):
    #     model_name = models[i]
    #     create_combined_graph(
    #         recall_50[i],
    #         recall_25[i],
    #         recall_75[i],
    #         baseline_recalls,
    #         [8, 16, 32],
    #         dataset_name + tfidf_features,
    #         "pickles/test_op_50/Hall"+model_name,
    #         recall_50[i][0].shape[0],
    #         20,
    #         tfidf_features,
    #     )

    # for i in range(len(starts)):
    #     fif = [
    #         recall_50[0][i],
    #         recall_50[1][i]
    #     ]
    #     seven = [
    #         recall_75[0][i],
    #         recall_75[1][i]
    #     ]
    #     twenty = [
    #         recall_25[0][i],
    #         recall_25[1][i]
    #     ]

    #     compare_models_graph(
    #         fif,
    #         twenty,
    #         seven,
    #         baseline_recalls,
    #         starts[i],
    #         dataset_name + tfidf_features,
    #         '',
    #         recall_50[0][0].shape[0],
    #         20,
    #         tfidf_features,
    #         models,
    #     )

def create_combined_graph(
    fifties,
    twenty_fives,
    seventy_fives,
    baseline_recalls,
    starts,
    filename,
    output_folder,
    iterations,
    no_statistical_validation,
    top_tfidf,
):
    plt.clf()
    sns.set_style("whitegrid")

    colors = ["royalblue", "orange", "green"]
    colorbands = ["skyblue", "gold", "mediumaquamarine"]

    for i, start in enumerate(starts):
        x_coor = range(start, len(twenty_fives[i]) + start)
        plt.plot(
            x_coor,
            fifties[i],
            label="50th percentile - Learner start at - " + str(start),
            linestyle="solid",
            color=colors[i],
        )
        plt.plot(
            x_coor,
            twenty_fives[i],
            label="25th percentile - Learner start at - " + str(start),
            linestyle="dotted",
            color=colorbands[i],
        )
        plt.plot(
            x_coor,
            seventy_fives[i],
            label="75th percentile - Learner start at - " + str(start),
            linestyle="dashed",
            color=colorbands[i],
        )

        plt.fill_between(
            x_coor, twenty_fives[i], seventy_fives[i], alpha=0.2, color=colorbands[i]
        )

        plt.axvline(x=start, color="black", linestyle="dotted", alpha=0.5)
        plt.text(start, 0.1, "Labeling Budget - " + str(start), rotation=90)

    plt.plot(baseline_recalls, label="Baseline", linestyle="dashdot", color="red")

    plt.xlabel(
        "Per iteration recall variance ("
        + str(iterations)
        + " iterations over "
        + str(no_statistical_validation)
        + " runs)"
    )
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.xscale("log")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.title("Learner Recalls with " + str(top_tfidf) + " TF-IDF features")
    plt.savefig(
        "combined_statistical_performance_" + filename + ".pdf",
        bbox_inches="tight",
    )
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
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.title('Comparing learners\' Recalls with '+str(top_tfidf)+' TF-IDF features')
    plt.savefig(output_folder+'compare_model_performance_start_'+str(start)+filename+'.pdf', bbox_inches='tight')
    return

if __name__ == "__main__":
    # Example usage
    plot_combined_stat_perf()
