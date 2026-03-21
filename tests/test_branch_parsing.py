from voice_ai_banking_support_agent.extraction.branch_parser import BranchParsingHints, parse_branch_records


def test_branch_parser_parses_table_rows_into_records():
    html = """
    <html><body>
      <table>
        <tr>
          <th>Մասնաճյուղ</th>
          <th>Քաղաք</th>
          <th>Հասցե</th>
          <th>Աշխ. ժամ</th>
          <th>Հեռ.</th>
        </tr>
        <tr>
          <td>Կենտրոն</td>
          <td>Երևան</td>
          <td>6 Northern Ave., Yerevan</td>
          <td>Երկ-Ուրբ 09:30-17:00</td>
          <td>+374 10 593333</td>
        </tr>
      </table>
    </body></html>
    """

    hints = BranchParsingHints(
        branch_name_keywords=["մասնաճյուղ", "branch"],
        city_keywords=["քաղաք", "city"],
        district_keywords=["թաղամաս", "district"],
        address_keywords=["հասցե", "address"],
        working_hours_keywords=["աշխ", "hours", "աշխ. ժամ", "working hours"],
        phone_keywords=["հեռ", "phone", "телефон"],
    )

    records = parse_branch_records(
        html,
        bank_name="Test Bank",
        source_url="https://example.invalid/branches",
        cleaned_text=None,
        hints=hints,
    )

    assert len(records) == 1
    r = records[0]
    assert r.branch_name == "Կենտրոն"
    assert r.city == "Երևան"
    assert "Northern" in r.address
    assert r.phone is not None
    assert "+374" in r.phone
    assert r.working_hours is not None

