# microcirculation
Цифровой прорыва 2022, микроциркуляция

## Установка зависимостей

```
pip install -r requirements.txt
```

## Обучение

```
python3 train.py /path/to/train/dataset /path/to/dir/for/checkpoints
```
где
- /path/to/train/dataset - путь до распакованного обучающего набора. Важно - при запуске скрипта обучения создадутся файлы .npz с масками и удалится одно изображение с пустой разметкой
- /path/to/dir/for/checkpoints - путь до директории, в которую сохранятся веса модели

## Выбор лучшей эпохи и копироване файлов с весами модели

```
python3 select_checkpoints.py /path/to/dir/for/checkpoints ./checkpoints
```

## Предсказание

```
python predict.py /path/to/train/dataset /path/to/dir/for/resulting/masks
```
где
- /path/to/train/dataset - путь до распакованного тестового набора
- /path/to/dir/for/checkpoints - путь до директории, в которую сохранятся получившиеся маски
