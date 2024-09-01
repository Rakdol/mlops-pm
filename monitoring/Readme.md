# Start Granfana and prometheus

```bash
$ docker compose -f grafana-docker-compose.yaml up -d
```

## Prometheus Connection Setting
![alt text](./images/image-2.png)
- promethus.yml 설정
- Docker bridge 주소로 localhost에 접속
```bash
# promethus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ["172.17.0.1:8000"] 
```
### metric 수집 
![alt text](./images/image-3.png)

## Source Postgres Connection Setting
![alt text](./images/image.png)

### Source data
![alt text](./images/image-4.png)

## Event Posgres Connection Setting
![alt text](./images/image-1.png)

### Event Data
![alt text](./images/image-5.png)