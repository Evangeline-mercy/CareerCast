from careercast import CareerCastClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"top_predictions": [{"career": "Data Scientist"}]}


class FakeSession:
    def __init__(self):
        self.call = None

    def request(self, method, url, **kwargs):
        self.call = (method, url, kwargs)
        return FakeResponse()


def test_predict_matches_fastapi_contract():
    session = FakeSession()
    client = CareerCastClient("https://example.test/", session=session, timeout=7)

    result = client.predict("Python, SQL", top_k=3)

    assert result["top_predictions"][0]["career"] == "Data Scientist"
    assert session.call == (
        "POST",
        "https://example.test/predict",
        {"timeout": 7, "json": {"skills_text": "Python, SQL", "top_k": 3}},
    )
