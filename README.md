docker build -t model_karimullin .

docker run --rm -v $(pwd)/artifacts:/python/artifacts model_karimullin

При исполнении, контейнер будет печатать логи с метриками в терминал. По окончанию обучения, артефакт модели будет сохранен на локальной машине в директории artifacts.
