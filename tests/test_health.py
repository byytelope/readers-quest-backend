import unittest

from fastapi.testclient import TestClient

from app.server import client


class TestHealthCheck(unittest.TestCase):
    test_client: TestClient = TestClient(client)

    def test_health_check(self):
        print("RUNNING TEST: Health")

        response = self.test_client.get("/health/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})
