# Предсказание исполнителя трека по набору акустических признаков

Описание задачи

На первый взгляд задача предсказания исполнителя трека выглядит странной, так как кажется, что эта информация изначально нам известна. Но при ближайшем рассмотрении, оказывается что не все так просто. Во-первых, есть задача разделения одноименных исполнителей. Когда к нам в каталог поступает новый релиз, то нам нужно как-то сопоставить исполнителя этого релиза с теми что уже есть в нашей базе и для одноименных исполнителей возникает неоднозначность. Во-вторых, и это менее очевидная часть, предсказывая исполнителей по аудио, мы неявным образом получаем модель которая выучивает похожесть исполнителей по звучанию и это также может быть полезным.

Формат входных данных

По лицензионным соглашениям мы не можем выкладывать исходные аудио треки, поэтому в рамках данной задачи мы решили подготовить для каждого трека признаковое описание на основе аудио сигнала. Изначально выбирается случайный фрагмент трека из центральной его части (70 процентов трека) длительностью 60 секунд, если трек короче 60 секунд, то берется трек целиком. Далее, этот фрагмент разбивается на чанки размером около 1.5 секунд и шагом порядка 740 миллисекунд и затем для каждого такого чанка аудио сигнала вычисляется вектор чисел, описывающий этот чанк, размером 512, это своего рода эмбединг этого чанка. Таким образом для каждого трека мы получаем последовательность векторов или другими словами матрицу размером 512xT сохраненную в файл в виде numpy array. Во входных данных задачи есть следующие файлы:

1. train_features.tar.gz
2. train_features_sample.tar.gz
3. test_features.tar.gz
4. train_meta.tsv
5. train_sample_meta.tsv
6. test_meta.tsv
7. compute_score.py
8. naive_baseline.py
9. nn_baseline.py

Первые два файла train_features.tar.gz и test_features.tar.gz это архивы с файлами, в которых хранятся признаковые описания треков обучающего и тестового подмножества соответственно.

Файл train_meta.tsv содержит отображение id треков в id исполнителей и дополнительно ссылку на относительный путь к файлу с признаковым описанием трека в архиве. Файл test_meta.tsv имеет аналогичный формат, за той лишь разницей что в нем нет id исполнителей треков. Мы отбирали треки таким образом, чтобы множества исполнителей в обучающем и тестовом подмножествах не пересекались.

Файл compute_score.py поможет вам посчитать метрику для вашего решения. Для простоты выделения валидационного подмножества треков, на котором можно оценивать метрику локально, мы разбили обучающее множество треков на 10 поддиректорий, внутри каждой из них исполнители треков также не пересекаются.

Файл naive_baseline.py содержит пример наивного решения, показывает как загружать из файла признаковые описания треков и как формировать файл с решением.

Файл nn_baseline.py содержит пример простейшего решения на основе нейросетей с использованием фреймворка pytorch

Кроме того мы добавили файлы train_features_sample.tar.gz и train_sample_meta.tsv которые содержат сэмпл данных для обучения

Все файлы можно скачать по ссылке: https://disk.yandex.ru/d/xKv1B88WtLZnPw

Альтернативно можно скачать данные (без скриптов) по ссылке https://storage.yandexcloud.net/audioml-contest22/dataset.tar.gz Сэмпл данных для обучения доступен по ссылке https://storage.yandexcloud.net/audioml-contest22/train_features_sample.tar.gz

Формат решения

Так как исполнители в обучающем и тестовом подмножествах разные, мы не можем подходить к решению этой задачи в лоб как к задаче классификации. Поэтому, задача заключается в том чтобы для каждого трека из тестового множества, по аудио признакам трека, построить отранжированный список остальных треков из тестового множества (исключая исходных трек-запрос для которого мы строим текущий список), такой, что чем выше трек в этом списке тем более вероятно что он принадлежит тому же исполнителю что и исходный трек-запрос. Для каждого трека из тестового множества нужно вывести одну строку в итоговый файл с решением. Формат строки query_trackid <tab> trackid1 <space> trackid2 … trackid100

Качество решения мы будем мерить с помощью метрики nDCG@100 (Normalized Discounted Cumulative Gain at K, K=100)

Ссылка на википедию: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

Во время соревнования в лидерборде будет отображаться максимальный public score из всех ваших валидных посылок. После завершения соревнования, лидерборд будет переранжирован согласно private score вашей последней валидной посылки. Кроме этого в лидерборде будет отображаться public score посчитанный также по вашему последнему валидному решению. Это означает что он может не совпадать (в частности быть ниже) с вашим максимальным public score

private nDCG: 0.5363996867993425
