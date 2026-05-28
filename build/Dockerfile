FROM registry.ceh19.onpp/base-go-builder:latest as build

WORKDIR /apps

COPY . .
ENV CGO_ENABLED=1
RUN go build -mod=readonly -o /app/detector ./cmd
RUN chmod +x /app/detector

COPY ./cmd/config.example.yaml /app/config.yaml

EXPOSE 8000
ENV TZ="Europe/Moscow"

ENTRYPOINT ["/app/detector", "FOREGROUND"]