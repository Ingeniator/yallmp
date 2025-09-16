# You can get base image list here https://git.sberdevices.ru/registry/tools
FROM registry.sberdevices.ru/registry/tools/redhat/ubi:latest as builder
WORKDIR /build
RUN make build


FROM registry.sberdevices.ru/registry/tools/redhat/ubi:latest
WORKDIR /app
COPY --from=builder /build .
EXPOSE 8080