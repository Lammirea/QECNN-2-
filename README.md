# Учебный проект, посвященный улучшению качества сжатых изображений при помощи нейронных сетей (модель QECNN)

## Датасет
Для обучения используется датасет изображений BSD500 [1]. Изображения из датасета сконвертированы в формат yuv и хранятся в едином файле:
при этом 400 изображений 480x320 находятся в файле BSD500train.yuv, и 100 изобажений находятся в файле BSD500test.yuv. Оба файла сжаты кодеком x265 с квантователем QP=35 и 
помещены в файлы BSD500train.yuv и BSD500test.yuv, соответственно.

## Результаты полученные после обучения на 10 эпохах на 100 изображениях
