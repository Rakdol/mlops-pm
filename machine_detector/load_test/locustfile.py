# locustfile.py

from locust import HttpUser, task, between


class OnnxUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_onnx(self):
        input_data = {"input_data": [[0.5] * 784]}
        self.client.post("/predict/", json=input_data)


class NonOnnxUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_non_onnx(self):
        input_data = {"input_data": [[0.5] * 784]}
        self.client.post("/predict/", json=input_data)


# # ONNX 서버에 대한 Locust 실행
# locust -f locustfile.py OnnxUser --host=http://localhost:8000

# # 비ONNX 서버에 대한 Locust 실행
# locust -f locustfile.py NonOnnxUser --host=http://localhost:8001
