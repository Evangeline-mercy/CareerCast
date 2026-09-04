from careercast import cli


def test_cli_predict_forwards_arguments(monkeypatch, capsys):
    class FakeClient:
        def __init__(self, base_url):
            assert base_url == "https://api.example"

        def predict(self, skills_text, *, top_k):
            assert skills_text == "Python, SQL"
            assert top_k == 2
            return {"top_predictions": []}

    monkeypatch.setattr(cli, "CareerCastClient", FakeClient)
    exit_code = cli.main(
        ["--api-url", "https://api.example", "predict", "Python, SQL", "--top-k", "2"]
    )

    assert exit_code == 0
    assert '"top_predictions": []' in capsys.readouterr().out
