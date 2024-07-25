FROM golang:1.18 AS builder
WORKDIR /build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o azure-oai-proxy .

RUN echo "#!/bin/sh\ncp /build/.env /app/.env 2>/dev/null || touch /app/.env" > /handle-env.sh
RUN chmod +x /handle-env.sh

FROM gcr.io/distroless/base-debian12
COPY --from=builder /build/azure-oai-proxy /
COPY --from=builder /app/.env /.env
EXPOSE 11437
ENTRYPOINT ["/azure-oai-proxy"]