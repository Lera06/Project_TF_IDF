import pandas as pd
import numpy as np
from stop_words import get_stop_words
import string
from django.shortcuts import render
from .forms import UploadForm
from .models import FileUpload


def upload_file(request):
    """Загружает файлы, формирует из них коллекцию документов, отображает результат"""

    all_files = FileUpload.objects.all()

    # Коллекция загруженных документов:
    original_corpus = []
    # Без знаков пунктуации:
    corpus_no_punct = []
    # В нижнем регистре:
    corpus_lowercase = []
    # Без стоп-слов:
    processed_corpus = []

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)

        for file in request.FILES.getlist('file'):
            original_corpus.append(file.read().decode())
        # print(original_corpus)
        for doc in original_corpus:
            corpus_no_punct.append(remove_punctuation(doc))
        # print(corpus_no_punct)
        for doc in corpus_no_punct:
            corpus_lowercase.append(get_lowercase(doc))
        # print(corpus_lowercase)
        for doc in corpus_lowercase:
            processed_corpus.append(remove_stop_words(doc))
        # print(processed_corpus)
        if form.is_valid():
            files = request.FILES.getlist('file')
            for file in files:
                 FileUpload.objects.create(file=file)
    else:
        form = UploadForm()

    contex = {
        'form': form,
        'all_files': all_files,
        'dataframe': get_dataframe(processed_corpus).head(50).to_html(),
        }

    return render(request, 'tf_idf/upload.html', contex)


# Вспомогательные методы для предварительной обработки документов:
def remove_punctuation(sentence: str) -> str:
    """
      Удаляет пунктуацию из предложения:
      :param sentence:
      :return: отформатированную строку
      """
    formatted_string = ''
    for chr in sentence:
        if chr not in string.punctuation:
            formatted_string += chr

    return formatted_string


def get_lowercase(sentence: str) -> str:
    """
    Переводит в нижний регистр каждый символ:
    :param sentence:
    :return: отформатированную строку
    """
    formatted_lst = []
    for word in sentence.split(' '):
        formatted_lst.append(word.lower())

    return ' '.join(formatted_lst)


def remove_stop_words(doc: str) -> str:
    """
    Удаляет стоп-слова:
    :param doc:
    :return: отформатированную строку
    """
    stop_words = get_stop_words('english')
    filtered_words = []

    for word in doc.split(' '):
        if word not in stop_words and word != '':
            filtered_words.append(word)

    return ' '.join(filtered_words)


def text_processing(docs: list) -> tuple:
    """
    Создает множество из уникальных слов:
    :param docs:
    :return: кортеж из множества, числа документов и количества уникальных слов:
    """
    # Определим коллекцию уникальных слов в уже предварительно обработанных документах:
    words_set = {word for doc in docs for word in doc.split(' ')}

    # Общее число документов в коллекции:
    number_of_docs = len(docs)
    # Количество уникальных слов в этой коллекции:
    number_unique_words = len(words_set)

    return words_set, number_of_docs, number_unique_words


# Вычисления:
def get_tf(docs: list) -> tuple:
    """
    Определяет TF (частоту слова в документе)
    :param docs:
    :return: кортеж из двух dataframes (транспонированной и оригинальной)
    """
    words_set = text_processing(docs)[0]
    number_of_docs = text_processing(docs)[1]
    number_unique_words = text_processing(docs)[2]

    # Создаем dataframe с нулевыми значениями: d = pd.DataFrame(np.zeros((N_rows, N_cols)))
    df_tf = pd.DataFrame(np.zeros((number_of_docs, number_unique_words)), columns=list(words_set))

    for i in range(number_of_docs):
        words = docs[i].split(' ')
        # Обращаемся к значениям в каждой ячейке по имени столбца - уникальному слову из коллекции
        for word in words:
            value = df_tf.loc[i, [word]] + (1 / len(words))
            df_tf.loc[i, [word]] = value

    df_tf_t = df_tf.copy()
    df_tf_t = df_tf_t.T
    # Возвращаем копию (df_tf_t) для транспонирования и оригинал (df_tf) для последующих вычислений
    return df_tf_t, df_tf


def get_idf(docs: list) -> tuple:
    """
    Определяет IDF (обратную частоту документа):
    :param docs:
    :return: кортеж из словаря и dataframe, созданного из этого словаря
    """

    words_set = text_processing(docs)[0]
    number_of_docs = text_processing(docs)[1]
    idf = {}

    for word in words_set:
        # Количество документов, которые содержат это слово
        count = 0

        for i in range(number_of_docs):
            if word in docs[i].split():
                # Кода данное слово присутствует в документе, отмечаем этот документ, увеличивая count на 1
                count += 1

        idf[word] = np.log10(number_of_docs / count)

    df_idf = pd.DataFrame.from_dict([idf])
    # Зададим новое имя для индекса, вместо нулевого значения:
    df_idf = df_idf.rename(index={0: 'idf'})
    df_idf = df_idf.T

    return idf, df_idf


def get_tf_idf(docs: list):
    """
    Определяет TF * IDF:
    :param docs:
    :return: dataframe
    """

    words_set = text_processing(docs)[0]
    number_of_docs = text_processing(docs)[1]
    df_tf = get_tf(docs)[1]
    idf = get_idf(docs)[0]
    df_tf_idf = df_tf.copy()

    for word in words_set:
        for i in range(number_of_docs):
            value = df_tf.loc[i, [word]] * idf[word]
            df_tf_idf.loc[i, [word]] = value

    df_tf_idf = df_tf_idf.T

    return df_tf_idf


def get_dataframe(docs: list):
    """
    Конкатенирует три dataframes, сортирует столбец idf по убыванию:
    :param docs:
    :return: dataframe со следующими столбцами:
        tf: значение tf в n-м количестве документов
        idf: значение idf по убыванию
        tf*idf: n-ом количестве документов
    """
    df_tf = get_tf(docs)[0]
    df_idf = get_idf(docs)[1]
    df_tf_idf = get_tf_idf(docs)

    # Объединяем все три dataframes по горизонтали:
    dataframe = pd.concat([df_tf, df_idf, df_tf_idf], axis=1)

    # Сортируем столбец idf по убыванию:
    dataframe = dataframe.sort_values(['idf'], ascending=False)

    return dataframe

