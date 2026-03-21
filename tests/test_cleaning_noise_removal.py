from voice_ai_banking_support_agent.extraction.cleaning import clean_html_to_text


def test_cleaning_removes_boilerplate_and_normalizes_whitespace():
    html = """
    <html>
      <body>
        <nav>Privacy Policy</nav>
        <header>ACBA</header>
        <p>Այս էջը վարկերի պայմանների մասին է։ Տոկոսադրույք՝ 12.5%:</p>
        <p>Գործող ժամկետ՝ 36 ամիս։ Մանրամասները ստորև։</p>
        <p>Վճարման եղանակները ներառում են ամսական անուիտետ։ Կիրառվում են լրացուցիչ պայմաններ և սահմանափակումներ։</p>
        <p>Տոկոսադրույքը կարող է փոփոխվել` կախված բանկի ներքին որոշումներից։ Խորհուրդ ենք տալիս ծանոթանալ պայմանագրին և փաստաթղթերին։</p>
        <p>Վարկի առավելագույն գումար՝ 5,000,000 դրամ։ Տոկոսադրույքի հաշվարկը կատարվում է համաձայն բանկի կանոնակարգերի։</p>
        <footer>All rights reserved</footer>
      </body>
    </html>
    """
    res = clean_html_to_text(html)
    assert res.usable is True
    assert "Privacy Policy" not in res.cleaned_text
    assert "All rights reserved" not in res.cleaned_text
    # Whitespace normalization: no double spaces.
    assert "  " not in res.cleaned_text

