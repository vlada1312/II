import sys
import numpy as np
import torch #библиотека для глубокого обучения.
# функциональный интерфейс torch.nn, содержащий полезные функции, такие как функции активации, потерь и пр.
import torch.nn.functional as F 

class Vocab: #утилиты для работы с вокабуляром токенов текста
    def __init__(self, tokens, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        """
        A special class that converts lines of tokens into matrices and backwards
        """
        #Проверка наличия специальных токенов в списке токенов.
        assert all(tok in tokens for tok in (bos, eos, unk))
        self.tokens = tokens
        self.token_to_ix = {t:i for i, t in enumerate(tokens)}
        self.bos, self.eos, self.unk = bos, eos, unk
        self.bos_ix = self.token_to_ix[bos]
        self.eos_ix = self.token_to_ix[eos]
        self.unk_ix = self.token_to_ix[unk]

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    # Статический метод для создания экземпляра Vocab из списка строк. 
    # Метод принимает строки, объединяет их, разделяет на токены, и создает упорядоченный
    def from_lines(lines, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        flat_lines = '\n'.join(list(lines)).split()
        tokens = sorted(set(flat_lines))
        tokens = [t for t in tokens if t not in (bos, eos, unk) and len(t)]
        tokens = [bos, eos, unk] + tokens
        return Vocab(tokens, bos, eos, unk)
    
    #Принимает строку и преобразует её в список токенов, добавляя к ней токены начала (bos) и конца (eos) последовательности
    #Если встречается токен, которого нет в словаре, он заменяется на токен unk (неизвестный).
    def tokenize(self, string):
        """converts string to a list of tokens"""
        tokens = [tok if tok in self.token_to_ix else self.unk
                  for tok in string.split()]
        return [self.bos] + tokens + [self.eos]

    # Преобразует список строк в матрицу идентификаторов токенов
    # Каждая строка токенизируется, и длина каждой последовательности дополняется до максимальной длины 
    # последовательности в пределах списка, используя индекс токена eos как заполнитель
    def to_matrix(self, lines, dtype=torch.int64, max_len=None):
        """
        convert variable length token sequences into  fixed size matrix
        example usage:
        >>>print(to_matrix(words[:3],source_to_ix))
        [[0 15 22 21 28 27 13 -1 -1 -1 -1 -1]
         [0 30 21 15 15 21 14 28 27 13 -1 -1]
         [0 25 37 31 34 21 20 37 21 28 19 13]]
        """
        lines = list(map(self.tokenize, lines))
        max_len = max_len or max(map(len, lines))

        matrix = torch.full((len(lines), max_len), self.eos_ix, dtype=dtype)
        for i, seq in enumerate(lines):
            row_ix = list(map(self.token_to_ix.get, seq))[:max_len]
            matrix[i, :len(row_ix)] = torch.as_tensor(row_ix)

        return matrix

    # Преобразует матрицу идентификаторов токенов обратно в список строк. 
    def to_lines(self, matrix, crop=True):
        """
        Convert matrix of token ids into strings
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        :param crop: if True, crops BOS and EOS from line
        """
        lines = []
        for line_ix in map(list,matrix):
            if crop:
                if line_ix[0] == self.bos_ix:
                    line_ix = line_ix[1:]
                if self.eos_ix in line_ix:
                    line_ix = line_ix[:line_ix.index(self.eos_ix)]
            line = ' '.join(self.tokens[i] for i in line_ix)
            lines.append(line)
        return lines
    
    # Создает маску для последовательностей токенов, где маска будет равна "1" (или True) для всех позиций до первого встреченного eos включительно.
    def compute_mask(self, input_ix):
        """ compute a boolean mask that equals "1" until first EOS (including that EOS) """
        return F.pad(torch.cumsum(input_ix == self.eos_ix, dim=-1)[..., :-1] < 1, pad=(1, 0, 0, 0), value=True)

