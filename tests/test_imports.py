import foundry_llm as fllm


def test_version():
    assert hasattr(fllm, "__version__")
