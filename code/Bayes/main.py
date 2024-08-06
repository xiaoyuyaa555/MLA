# list_of_posts, list_of_classes = bayes.loadDataSet()
# my_vocab_list = bayes.createVocabList(list_of_posts)
# print(my_vocab_list)
#
# re_vec1 = bayes.setOfWords2Vec(my_vocab_list, list_of_posts[0])
# re_vec2 = bayes.setOfWords2Vec(my_vocab_list, list_of_posts[1])
# print(re_vec1)
# print(re_vec2)

# list_of_posts, list_of_classes = bayes.loadDataSet()
# my_vocab_list = bayes.createVocabList(list_of_posts)
#
# train_mat = []
# for post in list_of_posts:
#     train_mat.append(bayes.setOfWords2Vec(my_vocab_list, post))
#
# p0_vec, p1_vec, p_ab = bayes.trainNB0(train_mat, list_of_classes)
# print(f"p_ab:{p_ab}")
# print(f"p0_vec:{p0_vec}")
# print(f"p1_vct:{p1_vec}")

# 0表示无侮辱词
# 1表示有侮辱词
# bayes.testingNB()
# from numpy import *
# import bayes
#
# bayes.spamTest()
import feedparser

from numpy import *
import feedparser
import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print(ny)
