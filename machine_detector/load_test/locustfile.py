# locustfile.py

from locust import HttpUser, task, between


class OnnxUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_onnx(self):
        input_data = {
            "data": [["L49624", "L ", 299.2, 308.6, 1267, 40.4, 76.0, 0, 0, 0, 0, 0]]
        }
        self.client.post("/v0.1/api/predict/", json=input_data)

# # ONNX 서버에 대한 Locust 실행
# locust -f locustfile.py OnnxUser --host=http://localhost:8000
