import ada_boost as ada


def load_data(file_path):
    with open(file_path, 'r') as f:
        num_feat = len(f.readline().split('\t'))

        data_mat = []
        label_mat = []
        for line in f.readlines():
            line_arr = []
            line = line.strip().split('\t')
            for i in range(num_feat - 1):
                line_arr.append(float(line[i]))
            data_mat.append(line_arr)
            if float(line[-1]) == 1.0:
                label_mat.append(1.0)
            else:
                label_mat.append(-1.0)
        return data_mat, label_mat


train_data_mat, train_label_mat = load_data('./horseColicTraining.txt')
test_data_mat, test_label_mat = load_data('./horseColicTest.txt')

for num_iter in [50]:
    classifier_list, agg_class_est = ada.ada_boost_train(train_data_mat, train_label_mat, num_iter)
    ada.plotROC(agg_class_est.T, train_label_mat)
    error_rate = ada.ada_test(test_data_mat, test_label_mat, classifier_list)
    print('%d个树桩的错误率:%.2f' % (num_iter,error_rate))

