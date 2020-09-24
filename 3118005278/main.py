# 正则包
import re,logging
# html 包
import html

from sys  import argv

# 自然语言处理包
import jieba
import jieba.analyse
# 机器学习包
from sklearn.metrics.pairwise import cosine_similarity


class Cosbesimilar(object):
    """
    余弦相似度
    """
    def __init__(self, content_x1, content_y2):
        
        self.file_one = content_x1
        self.file_second = content_y2

    @staticmethod
    def get_hot_decode(word_dict, key):  # oneHot编码
        cut_word = [0]*len(word_dict)
        for word in key:
            cut_word[word_dict[word]] += 1
        return cut_word

    @staticmethod
    def get_word(content):  # 提取关键词
        # 正则过滤 html 标签
        re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
        content = re_exp.sub(' ', content)
        # html 转义符实体化
        content = html.unescape(content)
        # 切割
        split_word = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keyList = jieba.analyse.extract_tags("|".join(split_word), topK=200, withWeight=False)
        return keyList


    def main(self):
        # 去除停用词
        jieba.analyse.set_stop_words('stopwords.txt')
        # 提取关键词
        key_one = self.get_word(self.file_one)
        key_second = self.get_word(self.file_second)
        # 词的并集
        union = set(key_one).union(set(key_second))
        # 编码
        word_dict = {}
        i = 0
        for word in union:
            word_dict[word] = i
            i += 1
        # oneHot编码
        cut_word_one = self.get_hot_decode(word_dict, key_one)
        cut_word_second = self.get_hot_decode(word_dict, key_second)
        # 余弦相似度计算
        sample = [cut_word_one, cut_word_second]
        # 除零处理
        try:
            sim = cosine_similarity(sample)
            return sim[1][0]
        except Exception as e:
            logging.error(e)
            return 0


def openfile(argv):
    '''
    打开文件对比操作
    '''
    f = open(argv[1],'r',encoding='utf-8')
    g = open(argv[2],'r',encoding='utf-8')
    answer = open(argv[3],'a+',encoding='utf-8')
    f1 = f.read()
    g1 = g.read()
    similarity = Cosbesimilar(f1, g1)
    similarity = similarity.main()
    strings = f'{argv[1]}和{argv[2]}相似度: %.2f%%' % (similarity*100)+"\n"
    answer.writelines(strings)
    print(strings)   
    f.close()
    g.close()
    answer.close()

# 测试
if __name__ == '__main__':
    
    openfile(argv)

