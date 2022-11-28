import argparse
import collections

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

def sentihood_strict_acc_4(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of Sentihood.
    """
    total_cases=int(len(y_true)/4)
    true_cases=0
    for i in range(total_cases):
        if y_true[i*4]!=y_pred[i*4]:continue
        if y_true[i*4+1]!=y_pred[i*4+1]:continue
        if y_true[i*4+2]!=y_pred[i*4+2]:continue
        if y_true[i*4+3]!=y_pred[i*4+3]:continue
        true_cases+=1
    aspect_strict_Acc = true_cases/total_cases

    return aspect_strict_Acc


def sentihood_macro_F1_4(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of Sentihood.
    """
    p_all=0
    r_all=0
    count=0
    for i in range(len(y_pred)//4):
        a=set()
        b=set()
        for j in range(4):
            if y_pred[i*4+j]!=0:
                a.add(j)
            if y_true[i*4+j]!=0:
                b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    # avoid zero division
    if Ma_p+Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)

    return aspect_Macro_F1


def sentihood_AUC_Acc_4(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Sentihood.
    Calculate "Acc" of sentiment classification task of Sentihood.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[]]
    aspect_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1) # "None": 1
        tmp_score=score[i][0] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%4].append(aspect_y_true[-1])
        aspect_y_scores[i%4].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(4):
        aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_score=[]
    sentiment_y_trues=[[],[],[],[]]
    sentiment_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            sentiment_y_true.append(y_true[i]-1) # "Postive":0, "Negative":1
            tmp_score=score[i][2]/(score[i][1]+score[i][2])  # probability of "Negative"
            sentiment_y_score.append(tmp_score)
            if tmp_score>0.5:
                sentiment_y_pred.append(1) # "Negative": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i%4].append(sentiment_y_true[-1])
            sentiment_y_scores[i%4].append(sentiment_y_score[-1])

    sentiment_auc=[]
    for i in range(4):
        sentiment_auc.append(metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i]))
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC


def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    s_all=0
    g_all=0
    s_g_all=0
    for i in range(len(y_pred)//5):
        s=set()
        g=set()
        for j in range(5):
            if y_pred[i*5+j]!=4:
                s.add(j)
            if y_true[i*5+j]!=4:
                g.add(j)
        if len(g)==0:continue
        s_g=s.intersection(g)
        s_all+=len(s)
        g_all+=len(g)
        s_g_all+=len(s_g)

    # avoid zero division
    if s_all == 0:
        p = 0.0
    else:
        p=s_g_all/s_all

    # avoid zero division
    if g_all == 0:
        r = 0.0
    else:
        r=s_g_all/g_all

    # avoid zero division
    if (p+r) == 0:
        f = 0.0
    else:
        f=2*p*r/(p+r)

    return p,r,f

def semeval_Acc(y_true, y_pred, score, classes=4):
    """
    Calculate "Acc" of sentiment classification task of SemEval-2014.
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."

    if classes == 4:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]==4:continue
            total+=1
            tmp=y_pred[i]
            if tmp==4:
                if score[i][0]>=score[i][1] and score[i][0]>=score[i][2] and score[i][0]>=score[i][3]:
                    tmp=0
                elif score[i][1]>=score[i][0] and score[i][1]>=score[i][2] and score[i][1]>=score[i][3]:
                    tmp=1
                elif score[i][2]>=score[i][0] and score[i][2]>=score[i][1] and score[i][2]>=score[i][3]:
                    tmp=2
                else:
                    tmp=3
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    elif classes == 3:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]>=3:continue
            total+=1
            tmp=y_pred[i]
            if tmp>=3:
                if score[i][0]>=score[i][1] and score[i][0]>=score[i][2]:
                    tmp=0
                elif score[i][1]>=score[i][0] and score[i][1]>=score[i][2]:
                    tmp=1
                else:
                    tmp=2
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    else:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]>=3 or y_true[i]==1:continue
            total+=1
            tmp=y_pred[i]
            if tmp>=3 or tmp==1:
                if score[i][0]>=score[i][2]:
                    tmp=0
                else:
                    tmp=2
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total

    return sentiment_Acc

def persent_strict_acc_7(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of PerSent with 7 aspects.
    """
    total_cases=int(len(y_true)/7)
    true_cases=0
    for i in range(total_cases):
        if y_true[i*7]!=y_pred[i*7]:continue
        if y_true[i*7+1]!=y_pred[i*7+1]:continue
        if y_true[i*7+2]!=y_pred[i*7+2]:continue
        if y_true[i*7+3]!=y_pred[i*7+3]:continue
        if y_true[i*7+4]!=y_pred[i*7+4]:continue
        if y_true[i*7+5]!=y_pred[i*7+5]:continue
        if y_true[i*7+6]!=y_pred[i*7+6]:continue
        true_cases+=1
    aspect_strict_Acc = true_cases/total_cases

    return aspect_strict_Acc

def persent_strict_acc_4(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of PerSent with 4 aspects.
    """
    total_cases=int(len(y_true)/4)
    true_cases=0
    for i in range(total_cases):
        if y_true[i*4]!=y_pred[i*4]:continue
        if y_true[i*4+1]!=y_pred[i*4+1]:continue
        if y_true[i*4+2]!=y_pred[i*4+2]:continue
        if y_true[i*4+3]!=y_pred[i*4+3]:continue
        true_cases+=1
    aspect_strict_Acc = true_cases/total_cases

    return aspect_strict_Acc

def persent_macro_F1_7(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of PerSent with 7 aspects.
    """
    p_all=0
    r_all=0
    count=0
    for i in range(len(y_pred)//7):
        a=set()
        b=set()
        for j in range(7):
            if y_pred[i*7+j]!=0:
                a.add(j)
            if y_true[i*7+j]!=0:
                b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    # avoid zero division
    if Ma_p+Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)

    return aspect_Macro_F1

def persent_macro_F1_4(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of PerSent with 4 aspects.
    """
    p_all=0
    r_all=0
    count=0
    for i in range(len(y_pred)//4):
        a=set()
        b=set()
        for j in range(4):
            if y_pred[i*4+j]!=0:
                a.add(j)
            if y_true[i*4+j]!=0:
                b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    # avoid zero division
    if Ma_p+Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)

    return aspect_Macro_F1

def persentV1_AUC_Acc_4(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of PerSent V1 with 4 labels and 4 aspects.
    Calculate "Acc" of sentiment classification task of PerSent V1 with 4 labels and 4 aspects.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[]]
    aspect_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1) # "None": 1
        tmp_score=score[i][0] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%4].append(aspect_y_true[-1])
        aspect_y_scores[i%4].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(4):
        try: 
            temp_auc = metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i])
        except ValueError:
            temp_auc = 0
        aspect_auc.append(temp_auc)
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_trues=[[],[],[],[]]
    sentiment_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            true_label=[0]*3
            true_label[y_true[i]-1]=1
            sentiment_y_true.append(y_true[i]-1) # "Neutral": 0, "Postive":1, "Negative":2
            tmp_score1=score[i][2]/(score[i][1]+score[i][2]+score[i][3])  # probability of "Positive"
            tmp_score2=score[i][3]/(score[i][1]+score[i][2]+score[i][3])  # probability of "Negative"
            tmp_score0=1-tmp_score1-tmp_score2
            tmp_scores=[tmp_score0, tmp_score1, tmp_score2]
            if tmp_score1>tmp_score2 and tmp_score1>tmp_score0:
                sentiment_y_pred.append(1) # "Positive": 1
            elif tmp_score2>tmp_score1 and tmp_score2>tmp_score0:
                sentiment_y_pred.append(2) # "Negative": 2
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i%4].append(true_label)
            sentiment_y_scores[i%4].append(tmp_scores)

    sentiment_auc=[]
    for i in range(4):
        try: 
            temp_auc1 = metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i])
        except ValueError:
            temp_auc1 = 0
        sentiment_auc.append(temp_auc1)
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def persentV1_AUC_Acc_7(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of PerSent V1 with 4 labels and 7 aspects.
    Calculate "Acc" of sentiment classification task of PerSent V1 with 4 labels and 7 aspects.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[],[],[],[]]
    aspect_y_scores=[[],[],[],[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1) # "None": 1
        tmp_score=score[i][0] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%7].append(aspect_y_true[-1])
        aspect_y_scores[i%7].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(7):
        try: 
            temp_auc = metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i])
        except ValueError:
            temp_auc = 0
        aspect_auc.append(temp_auc)
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_trues=[[],[],[],[],[],[],[]]
    sentiment_y_scores=[[],[],[],[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            true_label=[0]*3
            true_label[y_true[i]-1]=1
            sentiment_y_true.append(y_true[i]-1) # "Neutral": 0, "Postive":1, "Negative":2
            tmp_score1=score[i][2]/(score[i][1]+score[i][2]+score[i][3])  # probability of "Positive"
            tmp_score2=score[i][3]/(score[i][1]+score[i][2]+score[i][3])  # probability of "Negative"
            tmp_score0=1-tmp_score1-tmp_score2
            tmp_scores=[tmp_score0, tmp_score1, tmp_score2]
            if tmp_score1>tmp_score2 and tmp_score1>tmp_score0:
                sentiment_y_pred.append(1) # "Positive": 1
            elif tmp_score2>tmp_score1 and tmp_score2>tmp_score0:
                sentiment_y_pred.append(2) # "Negative": 2
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i%7].append(true_label)
            sentiment_y_scores[i%7].append(tmp_scores)

    sentiment_auc=[]
    for i in range(7):
        try: 
            temp_auc1 = metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i])
        except ValueError:
            temp_auc1 = 0
        sentiment_auc.append(temp_auc1)
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def persentV2_AUC_Acc_4(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of PerSent V2 with 3 labels and 4 aspects.
    Calculate "Acc" of sentiment classification task of PerSent V2 with 3 labels and 4 aspects.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[]]
    aspect_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1) # "None": 1
        tmp_score=score[i][0] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%4].append(aspect_y_true[-1])
        aspect_y_scores[i%4].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(4):
        try: 
            temp_auc = metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i])
        except ValueError:
            temp_auc = 0
        aspect_auc.append(temp_auc)
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_trues=[[],[],[],[]]
    sentiment_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            sentiment_y_true.append(y_true[i]-1) # "Negative":0, "Positive":1
            tmp_score=score[i][2]/(score[i][1]+score[i][2])  # probability of "Positive"
            sentiment_y_score.append(tmp_score)
            if tmp_score>0.5:
                sentiment_y_pred.append(1) # "Positive": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i%4].append(sentiment_y_true[-1])
            sentiment_y_scores[i%4].append(sentiment_y_score[-1])


    sentiment_auc=[]
    for i in range(4):
        try: 
            temp_auc1 = metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i])
        except ValueError:
            temp_auc1 = 0
        sentiment_auc.append(temp_auc1)
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def persentV2_AUC_Acc_7(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of PerSent V2 with 3 labels and 7 aspects.
    Calculate "Acc" of sentiment classification task of PerSent V2 with 3 labels and 7 aspects.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[],[],[],[]]
    aspect_y_scores=[[],[],[],[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1) # "None": 1
        tmp_score=score[i][0] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%7].append(aspect_y_true[-1])
        aspect_y_scores[i%7].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(7):
        try: 
            temp_auc = metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i])
        except ValueError:
            temp_auc = 0
        aspect_auc.append(temp_auc)
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_score=[]
    sentiment_y_trues=[[],[],[],[],[],[],[]]
    sentiment_y_scores=[[],[],[],[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            sentiment_y_true.append(y_true[i]-1) # "Negative":0, "Positive":1
            tmp_score=score[i][2]/(score[i][1]+score[i][2])  # probability of "Positive"
            sentiment_y_score.append(tmp_score)
            if tmp_score>0.5:
                sentiment_y_pred.append(1) # "Positive": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i%7].append(sentiment_y_true[-1])
            sentiment_y_scores[i%7].append(sentiment_y_score[-1])

    sentiment_auc=[]
    for i in range(7):
        try: 
            temp_auc1 = metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i])
        except ValueError:
            temp_auc1 = 0
        sentiment_auc.append(temp_auc1)
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def persentV1_strict_acc_long_4(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of PerSent V1 with 4 aspects on Longformer.
    """
    total_cases=int(len(y_true)/16)
    true_cases=0
    for i in range(total_cases):
        count = 0
        for j in range(16):
            if y_true[i*16+j]==y_pred[i*16+j]:count+=1
        if count == 16:true_cases+=1
    aspect_strict_Acc = true_cases/total_cases

    return aspect_strict_Acc

def persentV1_strict_acc_long_7(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of PerSent V1 with 7 aspects on Longformer.
    """
    total_cases=int(len(y_true)/28)
    true_cases=0
    for i in range(total_cases):
        count = 0
        for j in range(28):
            if y_true[i*28+j]==y_pred[i*28+j]:count+=1
        if count == 28:true_cases+=1
    aspect_strict_Acc = true_cases/total_cases

    return aspect_strict_Acc

def persentV1_macro_F1_long_4(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of PerSent V2 with 3 labels with 4 aspects.
    """
    p_all=0
    r_all=0
    count=0
    for i in range(len(y_pred)//16):
        a=set()
        b=set()
        for j in range(4):
            pred_count = 0
            true_count = 0
            for k in range(4):
                if k!=0 and y_pred[i*16+j*4+k]!=0: pred_count+=1
                if k!=0 and y_true[i*16+j*4+k]!=0: true_count+=1
            if pred_count==1: a.add(j)
            if true_count==1: b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    # avoid zero division
    if Ma_p+Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)

    return aspect_Macro_F1

def persentV1_macro_F1_long_7(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of PerSent V1 with 4 labels with 7 aspects.
    """
    p_all=0
    r_all=0
    count=0
    for i in range(len(y_pred)//28):
        a=set()
        b=set()
        for j in range(7):
            pred_count = 0
            true_count = 0
            for k in range(4):
                if k!=0 and y_pred[i*28+j*4+k]!=0: pred_count+=1
                if k!=0 and y_true[i*28+j*4+k]!=0: true_count+=1
            if pred_count==1: a.add(j)
            if true_count==1: b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    # avoid zero division
    if Ma_p+Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)

    return aspect_Macro_F1

def persentV1_AUC_Acc_long_4(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of PerSent V1 with 4 labels and 4 aspects.
    Calculate "Acc" of sentiment classification task of PerSent V1 with 4 labels and 4 aspects.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[],[],[],[]]
    aspect_y_scores=[[],[],[],[],[],[],[]]
    aspect_y_trues4=[[],[],[],[]]
    aspect_y_scores4=[[],[],[],[]]
    for i in range(len(y_true)//4):
        for j in range(4):
            if y_true[i*4+j]>0:
                if j == 0:
                    aspect_y_true.append(1) # "None": 1
                    break
                else:
                    aspect_y_true.append(0)
                    break
            
        tmp_score=score[i*4][1] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues4[i%4].append(aspect_y_true[-1])
        aspect_y_scores4[i%4].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(4):
        try: 
            temp_auc = metrics.roc_auc_score(aspect_y_trues4[i], aspect_y_scores4[i])
        except ValueError:
            temp_auc = 0
        aspect_auc.append(temp_auc)
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_trues=[[],[],[],[],[],[],[]]
    sentiment_y_scores=[[],[],[],[],[],[],[]]
    sentiment_y_trues4=[[],[],[],[]]
    sentiment_y_scores4=[[],[],[],[]]
    for i in range(len(y_true)//4):
        notNone = False
        true_label=[0]*3
        for j in range(4):
            if j != 0 and y_true[i*4+j] > 0:
                sentiment_y_true.append(j-1)
                true_label[j-1]=1
                notNone = True
                break
        if notNone:
            tmp_score1=score[i*3+1][1]/(score[i*3+1][1]+score[i*3+2][1]+score[i*3+3][1])  # probability of "Neutral"
            tmp_score2=score[i*3+2][1]/(score[i*3+1][1]+score[i*3+2][1]+score[i*3+3][1])  # probability of "Positive"
            tmp_score3=1-tmp_score1-tmp_score2
            tmp_scores=[tmp_score1, tmp_score2, tmp_score3]
            if tmp_score1>tmp_score2 and tmp_score1>tmp_score3:
                sentiment_y_pred.append(0) # "Neutral": 0
            elif tmp_score2>tmp_score1 and tmp_score2>tmp_score3:
                sentiment_y_pred.append(1) # "Positive": 1
            else:
                sentiment_y_pred.append(2)
            sentiment_y_trues4[i%4].append(true_label)
            sentiment_y_scores4[i%4].append(tmp_scores)

    sentiment_auc=[]
    for i in range(4):
        try: 
            temp_auc1 = metrics.roc_auc_score(sentiment_y_trues4[i], sentiment_y_scores4[i])
        except ValueError:
            temp_auc1 = 0
        sentiment_auc.append(temp_auc1)
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def persentV1_AUC_Acc_long_7(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of PerSent V1 with 4 labels and 7 aspects.
    Calculate "Acc" of sentiment classification task of PerSent V1 with 4 labels and 7 aspects.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[],[],[],[]]
    aspect_y_scores=[[],[],[],[],[],[],[]]
    aspect_y_trues4=[[],[],[],[]]
    aspect_y_scores4=[[],[],[],[]]
    for i in range(len(y_true)//4):
        for j in range(4):
            if y_true[i*4+j]>0:
                if j == 0:
                    aspect_y_true.append(1) # "None": 1
                    break
                else:
                    aspect_y_true.append(0)
                    break
            
        tmp_score=score[i*4][1] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%7].append(aspect_y_true[-1])
        aspect_y_scores[i%7].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(7):
        try: 
            temp_auc = metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i])
        except ValueError:
            temp_auc = 0
        aspect_auc.append(temp_auc)
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_trues=[[],[],[],[],[],[],[]]
    sentiment_y_scores=[[],[],[],[],[],[],[]]
    sentiment_y_trues4=[[],[],[],[]]
    sentiment_y_scores4=[[],[],[],[]]
    for i in range(len(y_true)//4):
        notNone = False
        true_label=[0]*3
        for j in range(4):
            if j != 0 and y_true[i*4+j] > 0:
                sentiment_y_true.append(j-1)
                true_label[j-1]=1
                notNone = True
                break
        if notNone:
            tmp_score1=score[i*3+1][1]/(score[i*3+1][1]+score[i*3+2][1]+score[i*3+3][1])  # probability of "Neutral"
            tmp_score2=score[i*3+2][1]/(score[i*3+1][1]+score[i*3+2][1]+score[i*3+3][1])  # probability of "Positive"
            tmp_score3=1-tmp_score1-tmp_score2
            tmp_scores=[tmp_score1, tmp_score2, tmp_score3]
            if tmp_score1>tmp_score2 and tmp_score1>tmp_score3:
                sentiment_y_pred.append(0) # "Neutral": 0
            elif tmp_score2>tmp_score1 and tmp_score2>tmp_score3:
                sentiment_y_pred.append(1) # "Positive": 1
            else:
                sentiment_y_pred.append(2)
            sentiment_y_trues[i%7].append(true_label)
            sentiment_y_scores[i%7].append(tmp_scores)
            
    sentiment_auc=[]
    for i in range(7):
        try: 
            temp_auc1 = metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i])
        except ValueError:
            temp_auc1 = 0
        sentiment_auc.append(temp_auc1)
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def persentV2_strict_acc_long_4(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of PerSent V2 with 4 aspects on Longformer.
    """
    total_cases=int(len(y_true)/12)
    true_cases=0
    for i in range(total_cases):
        count = 0
        for j in range(12):
            if y_true[i*12+j]==y_pred[i*12+j]:count+=1
        if count == 12:true_cases+=1
    aspect_strict_Acc = true_cases/total_cases

    return aspect_strict_Acc

def persentV2_strict_acc_long_7(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of PerSent V2 with 7 aspects on Longformer.
    """
    total_cases=int(len(y_true)/21)
    true_cases=0
    for i in range(total_cases):
        count = 0
        for j in range(21):
            if y_true[i*21+j]==y_pred[i*21+j]:count+=1
        if count == 21:true_cases+=1
    aspect_strict_Acc = true_cases/total_cases

    return aspect_strict_Acc

def persentV2_macro_F1_long_4(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of PerSent V2 with 3 labels with 4 aspects.
    """
    p_all=0
    r_all=0
    count=0
    for i in range(len(y_pred)//12):
        a=set()
        b=set()
        for j in range(4):
            pred_count = 0
            true_count = 0
            for k in range(3):
                if k!=0 and y_pred[i*12+j*3+k]!=0: pred_count+=1
                if k!=0 and y_true[i*12+j*3+k]!=0: true_count+=1
            if pred_count==1: a.add(j)
            if true_count==1: b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    # avoid zero division
    if Ma_p+Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)

    return aspect_Macro_F1

def persentV2_macro_F1_long_7(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of PerSent V2 with 3 labels with 7 aspects.
    """
    p_all=0
    r_all=0
    count=0
    for i in range(len(y_pred)//21):
        a=set()
        b=set()
        for j in range(7):
            pred_count = 0
            true_count = 0
            for k in range(3):
                if k!=0 and y_pred[i*21+j*3+k]!=0: pred_count+=1
                if k!=0 and y_true[i*21+j*3+k]!=0: true_count+=1
            if pred_count==1: a.add(j)
            if true_count==1: b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    # avoid zero division
    if Ma_p+Ma_r == 0:
        aspect_Macro_F1 = 0
    else:
        aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)

    return aspect_Macro_F1

def persentV2_AUC_Acc_long_4(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of PerSent V2 with 3 labels and 4 aspects.
    Calculate "Acc" of sentiment classification task of PerSent V2 with 3 labels and 4 aspects.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[],[],[],[]]
    aspect_y_scores=[[],[],[],[],[],[],[]]
    aspect_y_trues4=[[],[],[],[]]
    aspect_y_scores4=[[],[],[],[]]
    for i in range(len(y_true)//3):
        for j in range(3):
            if y_true[i*3+j]>0:
                if j == 0:
                    aspect_y_true.append(1) # "None": 1
                    break
                else:
                    aspect_y_true.append(0)
                    break
            
        tmp_score=score[i*3][1] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues4[i%4].append(aspect_y_true[-1])
        aspect_y_scores4[i%4].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(4):
        try: 
            temp_auc = metrics.roc_auc_score(aspect_y_trues4[i], aspect_y_scores4[i])
        except ValueError:
            temp_auc = 0
        aspect_auc.append(temp_auc)
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_trues=[[],[],[],[],[],[],[]]
    sentiment_y_scores=[[],[],[],[],[],[],[]]
    sentiment_y_trues4=[[],[],[],[]]
    sentiment_y_scores4=[[],[],[],[]]
    for i in range(len(y_true)//3):
        notNone = False
        true_label=[0]*2
        for j in range(3):
            if j != 0 and y_true[i*3+j] > 0:
                sentiment_y_true.append(j-1)
                true_label[j-1]=1
                notNone = True
                break
        if notNone:
            tmp_score=score[i*3+1][1]/(score[i*3+1][1]+score[i*3+2][1])  # probability of "Negative"
            tmp_scores = [tmp_score, 1-tmp_score]
            if tmp_score>0.5:
                sentiment_y_pred.append(0) # "Negative": 0
            else:
                sentiment_y_pred.append(1) # "Positive": 1
            sentiment_y_trues4[i%4].append(true_label)
            sentiment_y_scores4[i%4].append(tmp_scores)

    sentiment_auc=[]
    for i in range(4):
        try: 
            temp_auc1 = metrics.roc_auc_score(sentiment_y_trues4[i], sentiment_y_scores4[i])
        except ValueError:
            temp_auc1 = 0
        sentiment_auc.append(temp_auc1)
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def persentV2_AUC_Acc_long_7(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of PerSent V2 with 3 labels and 7 aspects.
    Calculate "Acc" of sentiment classification task of PerSent V2 with 3 labels and 7 aspects.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[],[],[],[]]
    aspect_y_scores=[[],[],[],[],[],[],[]]
    aspect_y_trues4=[[],[],[],[]]
    aspect_y_scores4=[[],[],[],[]]
    for i in range(len(y_true)//3):
        for j in range(3):
            if y_true[i*3+j]>0:
                if j == 0:
                    aspect_y_true.append(1) # "None": 1
                    break
                else:
                    aspect_y_true.append(0)
                    break
            
        tmp_score=score[i*3][1] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%7].append(aspect_y_true[-1])
        aspect_y_scores[i%7].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(7):
        try: 
            temp_auc = metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i])
        except ValueError:
            temp_auc = 0
        aspect_auc.append(temp_auc)
    aspect_Macro_AUC = np.mean(aspect_auc)
    
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_trues=[[],[],[],[],[],[],[]]
    sentiment_y_scores=[[],[],[],[],[],[],[]]
    sentiment_y_trues4=[[],[],[],[]]
    sentiment_y_scores4=[[],[],[],[]]
    for i in range(len(y_true)//3):
        notNone = False
        true_label=[0]*2
        for j in range(3):
            if j != 0 and y_true[i*3+j] > 0:
                sentiment_y_true.append(j-1)
                true_label[j-1]=1
                notNone = True
                break
        if notNone:
            tmp_score=score[i*3+1][1]/(score[i*3+1][1]+score[i*3+2][1])  # probability of "Negative"
            tmp_scores = [tmp_score, 1-tmp_score]
            if tmp_score>0.5:
                sentiment_y_pred.append(0) # "Negative": 0
            else:
                sentiment_y_pred.append(1) # "Positive": 1
            sentiment_y_trues[i%7].append(true_label)
            sentiment_y_scores[i%7].append(tmp_scores)
            
    sentiment_auc=[]
    for i in range(7):
        try: 
            temp_auc1 = metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i])
        except ValueError:
            temp_auc1 = 0
        sentiment_auc.append(temp_auc1)
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", \
                                "sentihood_NLI_B", "sentihood_QA_B", "semeval_single", \
                                "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to evalution.")
    parser.add_argument("--pred_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The pred data dir.")
    args = parser.parse_args()

    result = collections.OrderedDict()
    if args.task_name in ["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", "sentihood_NLI_B", "sentihood_QA_B"]:
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
        result = {'aspect_strict_Acc': aspect_strict_Acc,
                'aspect_Macro_F1': aspect_Macro_F1,
                'aspect_Macro_AUC': aspect_Macro_AUC,
                'sentiment_Acc': sentiment_Acc,
                'sentiment_Macro_AUC': sentiment_Macro_AUC}
    else:
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
        sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
        sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
        sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
        result = {'aspect_P': aspect_P,
                'aspect_R': aspect_R,
                'aspect_F': aspect_F,
                'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
                'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
                'sentiment_Acc_2_classes': sentiment_Acc_2_classes}

    for key in result.keys():
        print(key, "=",str(result[key]))
    

if __name__ == "__main__":
    main()
