.PHONY: build run tidy

# Сборка. OpenCV должен быть установлен (pkg-config --modversion opencv4).
build:
	go build -o bin/detector .

# Запуск с конфигом по умолчанию
run: build
	./bin/detector -config config.yaml

# Синхронизация зависимостей
tidy:
	go mod tidy

# Пример запроса к запущенному серверу
test-curl:
	curl -s -X POST http://localhost:8080/detect \
	     --data-binary @testdata/frame.jpg \
	     -H "Content-Type: image/jpeg" | jq .
