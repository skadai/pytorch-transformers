# -*- coding: utf-8 -*-

# @File    : data_preprocess.py
# @Date    : 2019-03-11
# @Author  : skym


import re
import emoji


def filter_nonprintable(text):
    """
    去除不可打印的字符
    :param text:
    :return:
    """
    import string
    # Get the difference of all ASCII characters from the set of printable
    # characters
    nonprintable = set([chr(i) for i in range(128)]
                       ).difference(string.printable)
    # Use translate to remove all non-printable characters
    return text.translate({ord(character): None for character in nonprintable})


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((0x9FFF >= cp >= 0x4E00) or
        (0x4DBF >= cp >= 0x3400) or
        (0x2A6DF >= cp >= 0x20000) or
        (0x2B73F >= cp >= 0x2A700) or
        (0x2B81F >= cp >= 0x2B740) or
        (0x2CEAF >= cp >= 0x2B820) or
        (0xFAFF >= cp >= 0xF900) or
        (0x2FA1F >= cp >= 0x2F800)):
        return True

    return False



def filter_irrelevant(text, keep_expression=False):
    """
    把链接和括号, 这些往往与情感分析无关
    # 不能删除, 可能是活动名
    """
    pat_http = re.compile(
        '(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?')

    if keep_expression:
        # 是否要保留expression
        pat_kh = re.compile('(@[^ ]+)[@|  ]|『[^』]+』|「[^」]+」|#[^#]+#|<[^>]+>')
    else:
        pat_kh = re.compile('&.{0,10};|\【[^】]{0,10}\】|\[[^\]]{0,10}\]|(@[^ ]+)[@| ]|『[^』]+』|「[^」]+」|#[^#]+#|<[^>]+>')

    patterns = [pat_http, pat_kh]
    ret = text
    for pat in patterns:
        ret = pat.sub('', ret)
    return ret.strip()


def prepare_text(text):
    """
        去除文本的emoji表情, 用空格替代换行
        :param text:
        :return:
    """

    clear_text = re.sub(r"\n", '', text)
    demojized_text = emoji.demojize(clear_text)
    ready_text = re.sub(':[\w_-]+:', '', demojized_text)
    ready_text = re.sub("[~@#￥%…&*（）；【】,。，！？]+", " ", ready_text)
    ready_text = re.sub('[ ]+', ',', ready_text)  # 替换空格为逗号
    return ready_text


def convert_text(content):
    """
    预处理文本
    :param content:
    :param brand:
    :return:
    """
    content = content.strip('\u200b')
    a = filter_irrelevant(content)
    a = filter_nonprintable(a)
    a = prepare_text(a)

    return a
