from pathlib import Path

from voice_ai_banking_support_agent.bank_manifest import load_banks_manifest


def test_manifest_loading_validates_and_loads_urls(tmp_path: Path):
    yaml_text = """
schema_version: "1"
banks:
  - bank_key: "acba"
    bank_name: "ACBA Bank"
    language: "hy"
    credits:
      urls:
        - "https://www.acba.am/en/individuals/loans/credit-lines"
    deposits:
      urls:
        - "https://www.acba.am/en/individuals/save-and-invest/deposits"
    branches:
      urls:
        - "https://www.acba.am/en/about-bank/Branches-and-ATMs"
"""
    p = tmp_path / "banks.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    manifest = load_banks_manifest(p)
    assert len(manifest.banks) == 1
    b = manifest.banks[0]
    assert b.bank_key == "acba"
    assert b.credits.urls
    assert b.deposits.urls
    assert b.branches.urls

