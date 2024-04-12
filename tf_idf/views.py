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
    corpus = []
    # Коллекция без знаков пунктуации:
    corpus_no_punct = []
    # Коллекция в нижнем регистре:
    processed_corpus = []

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)

        for file in request.FILES.getlist('file'):
            corpus.append(file.read().decode())

        for doc in corpus:
            corpus_no_punct.append(remove_punctuation(doc))

        for doc in corpus_no_punct:
            processed_corpus.append(get_lowercase(doc))

        if form.is_valid():
            files = request.FILES.getlist('file')
            for file in files:
                 FileUpload.objects.create(file=file)
    else:
        form = UploadForm()

    contex = {
        'form': form,
        'all_files': all_files,
        'tf': get_tf(processed_corpus).to_html(),
        'idf': get_idf(processed_corpus).to_html(),
        'tf_idf': get_tf_idf(processed_corpus).to_html(),
        }

    return render(request, 'tf_idf/upload.html', contex)


# Вспомогательные методы для предобработки документов:
def remove_punctuation(sentence):
    """Удаляет знаки пунктуации в предложении"""
    formatted_string = ''
    for chr in sentence:
        if chr not in string.punctuation:
            formatted_string += chr

    return formatted_string


def get_lowercase(sentence):
    formatted_lst = []
    for word in sentence.split(' '):
        formatted_lst.append(word.lower())

    return ' '.join(formatted_lst)


def text_processing(docs):
    """Создает коллекцию из уникальных слов"""
    # Определим коллекцию уникальных слов в коллекции уже предварительно обработанных документов:
    words_set = {word for doc in docs for word in doc.split(' ')}

    # Удаляем стоп-слова и получаем окончательный список уникальных слов:
    # stop_words = get_stop_words('english')
    # filtered_unique_words = [word for word in words_set if word not in stop_words]

    # Общее число документов в коллекции:
    number_of_docs = len(docs)
    # Количество уникальных слов в этой коллекции:
    number_unique_words = len(words_set)

    return words_set, number_of_docs, number_unique_words


# Вычисления:
def get_tf(docs):
    """Определяет TF (частота слова в документе)"""
    words_set = text_processing(docs)[0]
    number_of_docs = text_processing(docs)[1]
    number_unique_words = text_processing(docs)[2]

    # Создаем dataframe с нулевыми значениями: d = pd.DataFrame(np.zeros((N_rows, N_cols)))
    df_tf = pd.DataFrame(np.zeros((number_of_docs, number_unique_words)), columns=list(words_set))

    # Итерируемся по строкам (кол-во наших документов)
    for i in range(number_of_docs):
        words = docs[i].split(' ')
        # Обращаемся к значениям в каждой ячейке по имени столбца (соответствует уникальному слову из коллекции)
        for word in words:
            df_tf[word][i] = df_tf[word][i] + (1 / len(words))

    return df_tf


def get_idf(docs):
    """Определяет IDF (обратная частота документа)"""

    words_set = text_processing(docs)[0]
    number_of_docs = text_processing(docs)[1]
    idf = {}

    # Итерируемся по уникальным словам:
    for word in words_set:
        # Количество документов, которые содержат это слово
        count = 0

        # Число итераций соответствует количеству документов:
        for i in range(number_of_docs):
            if word in docs[i].split():
                # Если данное уникальное слово присутствует в документе, отмечаем этот документ, увеличивая count на 1
                count += 1

        idf[word] = np.log10(number_of_docs / count)

    df_idf = pd.DataFrame.from_dict([idf])

    return df_idf


def get_tf_idf(docs):
    """Определяет TF * IDF"""

    words_set = text_processing(docs)[0]
    number_of_docs = text_processing(docs)[1]
    df_tf = get_tf(docs)
    idf = get_idf(docs)
    df_tf_idf = df_tf.copy()

    for word in words_set:
        for i in range(number_of_docs):
            df_tf_idf[word][i] = df_tf[word][i] * idf[word]

    return df_tf_idf

