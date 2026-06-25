# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import json
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, average_precision_score,  precision_recall_curve, matthews_corrcoef

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy==1]) / len(testy)
    #yhat = np.array(pred)

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    #lr_f1 = f1_score(testy, yhat)
    #print(type(lr_precision), type(lr_recall))
    #print(np.shape(lr_precision), np.shape(lr_recall))
    lr_auc = auc(lr_recall, lr_precision)
    # summarize scores
    #print('AUC-PR:  auc=%.3f' % ( lr_auc))
    # plot the precision-recall curves
    return  lr_auc
    
def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx1,idx2,label=line.split()
            answers[(idx1,idx2)]=int(label)
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx1,idx2,label=line.split()
            predictions[(idx1,idx2)]=int(label)
    return predictions

def calculate_scores(answers,predictions):
    y_trues,y_preds=[],[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for ({},{}) pair.".format(key[0],key[1]))
            sys.exit()
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
    scores={}
    scores['Accuracy']=accuracy_score(y_trues, y_preds)
    scores['Precision']=precision_score(y_trues, y_preds)
    scores['Recall']=recall_score(y_trues, y_preds)
    scores['F1']=f1_score(y_trues, y_preds)
    scores['MCC']=matthews_corrcoef(y_trues, y_preds)
    scores['AUC-ROC']=roc_auc_score(y_trues, y_preds)
    scores['AUC-PR']=auc_pc(y_trues, y_preds)
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for BigCloneBench dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    # 打开文件以追加模式写入（如果文件不存在则创建）
    file_path = 'performance.txt'
    file = open(file_path, 'a')

    # 写入内容
    file.write(json.dumps(scores))
    file.write('\n')

    # 关闭文件
    file.close()

    
    print(scores)

if __name__ == '__main__':
    main()
