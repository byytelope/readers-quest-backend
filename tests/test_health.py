import unittest

from fastapi.testclient import TestClient

from app.server import client
from utils.log import Logger


class TestHealthCheck(unittest.TestCase):
    test_client = TestClient(client)

    def test_health_check(self):
        Logger.info("RUNNING TEST: Health")

        response = self.test_client.get("/health/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})