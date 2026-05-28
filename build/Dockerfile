FROM registry.ceh19.onpp/base-go-builder:latest AS build

WORKDIR /apps

COPY go.mod go.sum ./
RUN go mod download

COPY . .

ENV CGO_ENABLED=1
RUN go build -mod=readonly -o /app/detector ./cmd

# ------------------------------------------------------------------
FROM registry.ceh19.onpp/base-go-builder:latest AS runtime

WORKDIR /app

COPY --from=build /app/detector ./detector
COPY ./cmd/config.example.yaml ./config.yaml

# Папка с файлами модели (yolov4-tiny.weights, yolov4-tiny.cfg, coco.names).
# Файлы должны лежать в ./models/ рядом с Dockerfile перед сборкой.
# Они не коммитятся в git (.gitignore), скачиваются отдельно.
COPY ./models ./models

EXPOSE 8000
ENV TZ="Europe/Moscow"

ENTRYPOINT ["/app/detector"]
