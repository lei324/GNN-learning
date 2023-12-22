# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    eval_utils.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/22 17:14:03 by lei324            #+#    #+#              #
#    Updated: 2023/12/22 17:14:03 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from sklearn.metrics import average_precision_score, accuracy_score, f1_score


def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1
