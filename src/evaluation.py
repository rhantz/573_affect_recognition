import sys
import os

to_round = 4


def read_gold(filename):
    lines = filename.readlines()
    result = []
    for x in lines:
        result.append(x.rstrip().split("\t"))
    filename.close()
    return result


def read_predictions(filename):
    lines = filename.readlines()
    result = [line.strip() for line in lines]
    return result


def calculatePRF(gold, prediction):
    """
	gold/prediction list of list of emo predictions 
	"""
    # initialize counters
    labels = set(gold + prediction)
    tp = dict.fromkeys(labels, 0.0)
    fp = dict.fromkeys(labels, 0.0)
    fn = dict.fromkeys(labels, 0.0)
    precision = dict.fromkeys(labels, 0.0)
    recall = dict.fromkeys(labels, 0.0)
    f = dict.fromkeys(labels, 0.0)
    # check every element
    for g, p in zip(gold, prediction):
        # TP
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
            # print("Label\tTP\tFP\tFN\tP\tR\tF")
    for label in labels:
        recall[label] = (
            0.0
            if (tp[label] + fn[label]) == 0.0
            else (tp[label]) / (tp[label] + fn[label])
        )
        precision[label] = (
            1.0
            if (tp[label] + fp[label]) == 0.0
            else (tp[label]) / (tp[label] + fp[label])
        )
        f[label] = (
            0.0
            if (precision[label] + recall[label]) == 0
            else (2 * precision[label] * recall[label])
            / (precision[label] + recall[label])
        )
        microrecall = (sum(tp.values())) / (sum(tp.values()) + sum(fn.values()))
        microprecision = (sum(tp.values())) / (sum(tp.values()) + sum(fp.values()))
        microf = (
            0.0
            if (microprecision + microrecall) == 0
            else (2 * microprecision * microrecall) / (microprecision + microrecall)
        )
    # Macro average
    macrorecall = sum(recall.values()) / len(recall)
    macroprecision = sum(precision.values()) / len(precision)
    macroF = sum(f.values()) / len(f)

    accuracy = 0
    for label in labels:
        accuracy += tp[label]

    accuracy = accuracy / len(gold)

    return (
        round(microrecall, to_round),
        round(microprecision, to_round),
        round(microf, to_round),
        round(macrorecall, to_round),
        round(macroprecision, to_round),
        round(macroF, to_round),
        round(accuracy, to_round),
    )


def read_file(submission_path, nb_labels=2, nb_samp=10):
    """
	Read the tsv file
	"""
    # unzipped submission data is always in the 'res' subdirectory
    if not os.path.exists(submission_path):
        print("Could not find submission file {0}".format(submission_path))
        predictedList = [[0] * nb_labels] * nb_samp
    else:
        submission_file = open(os.path.join(submission_path))
        predictedList = read_predictions(submission_file)

    return predictedList


def score(ref_path, res_path, output_dir):
    truth_file = open(ref_path)
    goldList = read_gold(truth_file)
    gold_emo = [k[2] for k in goldList]
    nb_samp = len(goldList)

    predictedList = read_file(submission_path=res_path, nb_labels=1, nb_samp=nb_samp)

    if len(goldList) != len(predictedList):
        print("Number of labels is not aligned!")
        sys.exit(1)

    (micror, microp, microf, macror, macrop, macrof, accuracy) = calculatePRF(
        gold_emo, predictedList
    )

    with open(os.path.join(output_dir, "scores.txt"), "w") as output_file:
        str_to_write = "Macro F1-Score: {5}\nMicro Recall: {0}\nMicro Precision: {1}\nMicro F1-Score: {2}\nMacro Recall: {3}\nMacro Precision: {4}\nAccuracy: {6}\n".format(
            micror, microp, microf, macror, macrop, macrof, accuracy
        )
        output_file.write(str_to_write)


# [_, ref_path, res_path, output_dir] = sys.argv
# ref_path = "../data/eng/dev/goldstandard_dev_2022.tsv"
# res_path = "../outputs/predictions_EMO.tsv"
# output_dir = "../outputs"


# if __name__ == "__main__":
#     score(ref_path, res_path, output_dir)

