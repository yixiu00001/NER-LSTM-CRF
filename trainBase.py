# coding: utf-8

from data import *#read_corpus, read_dictionary, tag2label, random_embedding,vocab_build,batch_yield,batch_yieldOneIn,pad_sequences
import sys, time
from tensorflow.contrib.crf import viterbi_decode
from eval import conlleval


#数据填充，先将同一batch的数据pad成相同长度
def get_feed_dict(model, seqs, labels = None, lr = None, dropout = None):
    word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
    feed_dict = {model.word_ids:word_ids,
                model.sequence_lengths : seq_len_list}
    if labels is not None:
        labels_, _ = pad_sequences(labels, pad_mark=0)
        feed_dict[model.labels] = labels_
    if lr is not None:
        feed_dict[model.lr_pl ] = lr
    if dropout is not None:
        feed_dict[model.dropout_pl] = dropout

    return feed_dict, seq_len_list
#一个迭代的训练
'''
1.将字映射成index
2.shuffle后迭代训练每一批数据
3.计算loss，更新参数等
4.最后评估测试集合效果：计算预测结果的index，转化成tag，和样本中的tag比对
'''
def run_one_epoch(model, sess, train, dev, tag2label,epoch, saver):
    num_batches = (len(train)+model.batch_size-1)//model.batch_size
    model.logger.info("train lenght={} number_batches={}".format(len(train), num_batches))

    #start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #载入的训练集合将字映射成index
    train_ = raw2Index(train,model.vocab )
    batches = batch_yield(train_, model.batch_size, model.vocab, model.tag2label,shuffle=model.shuffle)
    start_time0 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    model.logger.info("=========={} epoch begin train, time is {}".format(epoch+1, start_time0))
    for step ,(seq, labels) in enumerate(batches):
        nums=0
        for i in range(len(seq)):
            nums = nums + len(seq[i])
        #model.logger.info("======seq length======{}======all length={}".format(len(seq), nums))
        sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
        step_num = epoch*num_batches + step +1
        feed_dict, _ = get_feed_dict(model,seq, labels, model.lr, model.dropout_keep_prob)

        _, loss_train, summary, step_num_=  sess.run([model.train_op, model.loss, model.merged, model.global_step],
                                                  feed_dict=feed_dict)
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if step+1 ==1 or (step+1) %10==0 or step+1 ==num_batches:
            model.logger.info(
                '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                loss_train, step_num))
            #model.logger.info("=============validation==========")
            #label_list_dev , seq_len_list_dev = dev_one_epoch(model,sess, dev)
            #evaluate(model,label_list_dev, seq_len_list_dev, dev, epoch)

        model.file_writer.add_summary(summary, step_num)

        
    model.logger.info("=============validation==========")
    label_list_dev , seq_len_list_dev = dev_one_epoch(model,sess, dev)
    evaluate(model,label_list_dev, seq_len_list_dev, dev, epoch)
#一个迭代的测试
def dev_one_epoch(model, sess, dev ):

    label_list , seq_len_list = [], []
    #seq_list = []
    #直接load的数据集合是文本词，先转换成index
    dev_ = raw2Index(dev,model.vocab )
    #print("dev=", len(dev), "dev_=", len(dev_))
    for seqs, labels in batch_yield(dev_, model.batch_size, model.vocab,model.tag2label, shuffle=False):
        #print("seq[0]=",seqs[0])
        label_list_, seq_len_list_ = predict_one_batch(model,sess, seqs)
        #print("len=",len(label_list_))
        label_list.extend(label_list_)
        seq_len_list.extend(seq_len_list_)
        #seq_list.extend(seqs)

    return label_list, seq_len_list#,  seq_list
#一个batch的预测，如果是CRF，则调用维特比方法解码
def predict_one_batch(model, sess, seqs):
    feed_dict, seq_len_list = get_feed_dict(model,seqs, dropout=1.0)

    if model.CRF:
        logits, transition_params = sess.run(
            [model.logits, model.transition_params],
            feed_dict = feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list, seq_len_list
    else:
        label_list = sess.run(model.labels_softmax_, feed_dict=feed_dict)
        return label_list, seq_len_list
#准确等评估，使用已有的perl直接产出总体和每类实体的准确/精准/召回和F-score 
#注意：将预测的label映射成tag，即B I O格式
def evaluate(model, label_list, seq_len_list, data, epoch=None):
    label2tag = {}
    for tag, label in model.tag2label.items():
        label2tag[label] = tag if label!=0 else label
    #将预测出的label index映射成label，因为perl文件中统计的是B I 等开头的标签
    model_predict = []
    for label_, (sent, tag) in zip(label_list, data):
        tag_ = [label2tag[label__] for label__ in label_]
        sent_res = []
        if len(label_)!=len(sent):
            print("len=",len(sent), len(label_),len(tag_))
        for i in range(len(sent)):
            sent_res.append([sent[i], tag[i], tag_[i]])
        model_predict.append(sent_res)

        #print(model_predict)

        
    epoch_num = str(epoch+1) if epoch!=None else 'test'
    label_path = os.path.join(model.result_path, 'label_'+epoch_num)
    metric_path = os.path.join(model.result_path, 'result_metric_' + epoch_num)

    for item in conlleval(model_predict, label_path, metric_path):
        print(item)
#输入文本，输出实体检测结果        
def demo_one(model, sess, demo_data):
    label_list = []
    
    for seqs, labels in batch_yieldOneIn(demo_data, model.batch_size,model.vocab,model.tag2label, shuffle=False):
        label_list_, _ = predict_one_batch(model, sess, seqs)
        label_list.extend(label_list_)

    #print(label_list)
    label2tag = {}
    for tag, label in model.tag2label.items():
        label2tag[label] = tag if label!=0 else label
    tag = [label2tag[label] for label in label_list[0]]

    #print( tag)
    return tag